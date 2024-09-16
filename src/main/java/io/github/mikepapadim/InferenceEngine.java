package io.github.mikepapadim;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

import io.github.mikepapadim.gpu.shared.MemObject;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroKernel;

/**
 * This class performs forward inference using a Transformer model.
 */
public class InferenceEngine {

    /**
     * Performs forward inference using the Transformer model.
     *
     * @param transformer
     *            The Transformer model to use for inference.
     * @param token
     *            The input token for processing.
     * @param pos
     *            The position of the input token in the sequence.
     * @param executionPlan
     *            The list of TornadoExecutionPlan objects for execution.
     * @return The output logits produced by the Transformer model.
     */
    static MemorySegment forwardWithTornadoVM(Transformer transformer, int token, int pos, TornadoExecutionPlan executionPlan) {
        Config p = transformer.config;
        WeightsFP32 w = (WeightsFP32) transformer.weights;
        RunStateFloat s = (RunStateFloat) transformer.state;
        int dim = p.dim;
        int hidden_dim = p.hidden_dim;
        int head_size = p.head_size;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        // copy the token embedding into x
        // w.token_embedding_table.get(token * dim, s.x, 0, dim);
        FloatBuffer auxW = w.token_embedding_table;
        int kk = 0;
        for (int i = token * dim; i < (token * dim) + dim; i++) {
            s.x.set(kk++, auxW.get(i));
        }

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {

            // attention rmsnorm
            rmsnorm(s.xb, s.x, w.rms_att_weight[l], dim);

            // qkv matmuls for this position
            MatrixVectorCollection.matmul(s.q, s.xb, w.wq[l], dim, dim);
            MatrixVectorCollection.matmul(s.k, s.xb, w.wk[l], dim, kv_dim);
            MatrixVectorCollection.matmul(s.v, s.xb, w.wv[l], dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq = (float) (1.0 / Math.pow(10000.0f, head_dim / (float) head_size));
                float val = pos * freq;
                float fcr = (float) Math.cos(val);
                float fci = (float) Math.sin(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    FloatArray vec = v == 0 ? s.q : s.k; // the vector to rotate (query or key)
                    float v0 = vec.get(i);
                    float v1 = vec.get(i + 1);
                    vec.set(i, (v0 * fcr - v1 * fci));
                    vec.set(i + 1, (v0 * fci + v1 * fcr));
                }
            }

            // save key,value at this time step (pos) to our kv cache
            // int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
            //System.arraycopy(s.k, 0, s.key_cache[l], pos * kv_dim, kv_dim);
            //System.arraycopy(s.v, 0, s.value_cache[l], pos * kv_dim, kv_dim);
            copyMemObjects(s.k, 0, s.key_cache[l], pos * kv_dim, kv_dim);
            copyMemObjects(s.v, 0, s.value_cache[l], pos * kv_dim, kv_dim);


            final int curLayer = l;

            // multihead attention. iterate over all heads
            IntStream.range(0, p.n_heads).parallel().forEach(h -> {
                // get the query vector for this head
                // float* q = s.q + h * head_size;
                int qOffset = h * head_size;

                // attention scores for this head
                // float* att = s.att + h * p.seq_len;
                int attOffset = h * p.seq_len;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int keyCacheOffset = t * kv_dim + (h / kv_mul) * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        //score += s.q[qOffset + i] * s.key_cache[curLayer][keyCacheOffset + i];
                        score += s.q.get(qOffset + i) * s.key_cache[curLayer][keyCacheOffset + i];
                    }
                    score /= (float) Math.sqrt(head_size);
                    // save the score to the attention buffer
                    s.att[attOffset + t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(s.att, attOffset, pos + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * head_size;
                int xbOffset = h * head_size;
                // memset(xb, 0, head_size * sizeof(float));
                //Arrays.fill(s.xb, xbOffset, xbOffset + head_size, 0f);
                segmentFill(s.xb, xbOffset, xbOffset + head_size, 0f);

                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int vOffset = t * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = s.att[attOffset + t];
                    // accumulate the weighted value inconfigto xb
                    for (int i = 0; i < head_size; i++) {
                        float acc = s.xb.get(xbOffset + i);
                        s.xb.set(xbOffset + i, (acc + (a * s.value_cache[curLayer][vOffset + i])));
                    }
                }
            });

            // final matmul to get the output of the attention
            MatrixVectorCollection.matmul(s.xb2, s.xb, w.wo[l], dim, dim);

            residualConnection(s.x, s.xb2, dim);

            // ffn rmsnorm
            rmsnorm(s.xb, s.x, w.rms_ffn_weight[l], dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            MatrixVectorCollection.matmul(s.hb, s.xb, w.w1[l], dim, p.hidden_dim);
            MatrixVectorCollection.matmul(s.hb2, s.xb, w.w3[l], dim, p.hidden_dim);

            fusedSiluEwiseMul(hidden_dim, s.hb, s.hb2);

            // final matmul to get the output of the ffn
            MatrixVectorCollection.matmul(s.xb, s.hb, w.w2[l], p.hidden_dim, dim);

            residualConnection(s.x, s.xb, dim);
        }

        // final rmsnorm
        rmsnorm(s.x, s.x, w.rms_final_weight, dim);
        // MatrixVectorCollection.matmul(s.logits, s.x, w.wcls, dim, p.vocab_size);
        // MatrixVectorCollection.matmul(s.logits, s.x, w.weightTensor, dim,

        // invoke TornadoVM to run MatMul on the GPU
        TornadoDevice device = TornadoExecutionPlan.getDevice(0, 1);
        executionPlan //
                .withDevice(device) //
                .execute(); //

        return s.logits.getSegment();
    }

    private static void copyMemObjects(MemObject source, final int index, float[] dest, final int destIndex, final int length) {
        int j = destIndex;
        for (int i = index; i < length; i++) {
            dest[j++] = source.get(i);
        }
    }

    private static void copyMemObjects(FloatArray source, final int index, float[] dest, final int destIndex, final int length) {
        int j = destIndex;
        for (int i = index; i < length; i++) {
            dest[j++] = source.get(i);
        }
    }

    private static void segmentFill(MemObject memObject, int fromIndex, int toIndex, float value) {
        for (int i = fromIndex; i < toIndex; i++)
            memObject.set(i, value);
    }

    private static void segmentFill(FloatArray memObject, int fromIndex, int toIndex, float value) {
        for (int i = fromIndex; i < toIndex; i++)
            memObject.set(i, value);
    }

    static MemorySegment forwardWithLevelZero(Transformer transformer, int token, int pos) {
        Config p = transformer.config;
        WeightsShared w = (WeightsShared) transformer.weights;
        RunStateSharedMem s = (RunStateSharedMem) transformer.state;
        LevelZeroKernel kernel = s.matMulkernel;
        int dim = p.dim;
        int hidden_dim = p.hidden_dim;
        int head_size = p.head_size;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        // copy the token embedding into x
        // w.token_embedding_table.get(token * dim, s.x, 0, dim);
        MemObject auxW = w.token_embedding_table;
        int kk = 0;
        for (int i = token * dim; i < (token * dim) + dim; i++) {
            s.x.set(kk++, auxW.get(i));
        }

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {

            // attention rmsnorm
            rmsnorm(s.xb, s.x, w.rms_att_weight[l], dim);

            // qkv matmuls for this position
            MatrixVectorCollection.matmul(s.q, s.xb, w.wq[l], dim, dim);
            MatrixVectorCollection.matmul(s.k, s.xb, w.wk[l], dim, kv_dim);
            MatrixVectorCollection.matmul(s.v, s.xb, w.wv[l], dim, kv_dim);


            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq = (float) (1.0 / Math.pow(10000.0f, head_dim / (float) head_size));
                float val = pos * freq;
                float fcr = (float) Math.cos(val);
                float fci = (float) Math.sin(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    MemObject vec = v == 0 ? s.q : s.k; // the vector to rotate (query or key)
                    float v0 = vec.get(i);
                    float v1 = vec.get(i + 1);
                    vec.set(i, v0 * fcr - v1 * fci);
                    vec.set(i + 1 , v0 * fci + v1 * fcr);
                }
            }

            // save key,value at this time step (pos) to our kv cache
            // int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience

            // Copies
            //            System.arraycopy(s.k, 0, s.key_cache[l], pos * kv_dim, kv_dim);
            //            System.arraycopy(s.v, 0, s.value_cache[l], pos * kv_dim, kv_dim)
            copyMemObjects(s.k, 0, s.key_cache[l], pos * kv_dim, kv_dim);
            copyMemObjects(s.v, 0, s.value_cache[l],  pos * kv_dim, kv_dim);

            final int curLayer = l;

            // multihead attention. iterate over all heads
            IntStream.range(0, p.n_heads).parallel().forEach(h -> {
                // get the query vector for this head
                // float* q = s.q + h * head_size;
                int qOffset = h * head_size;

                // attention scores for this head
                // float* att = s.att + h * p.seq_len;
                int attOffset = h * p.seq_len;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int keyCacheOffset = t * kv_dim + (h / kv_mul) * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += s.q.get(qOffset + i) * s.key_cache[curLayer][keyCacheOffset + i];
                    }
                    score /= (float) Math.sqrt(head_size);
                    // save the score to the attention buffer
                    s.att[attOffset + t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(s.att, attOffset, pos + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * head_size;
                int xbOffset = h * head_size;
                // memset(xb, 0, head_size * sizeof(float));
                // Arrays.fill(s.xb, xbOffset, xbOffset + head_size, 0f);
                segmentFill(s.xb, xbOffset, xbOffset + head_size, 0f);

                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int vOffset = t * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = s.att[attOffset + t];
                    // accumulate the weighted value inconfigto xb
                    for (int i = 0; i < head_size; i++) {
                        //s.xb[xbOffset + i] += a * s.value_cache[curLayer][vOffset + i];
                        float acc = s.xb.get(xbOffset + i);
                        s.xb.set(xbOffset + i, (acc + (a * s.value_cache[curLayer][vOffset + i])));
                    }
                }
            });

            // final matmul to get the output of the attention
            MatrixVectorCollection.matmul(s.xb2, s.xb, w.wo[l], dim, dim);

            residualConnection(s.x, s.xb2, dim);

            // ffn rmsnorm
            rmsnorm(s.xb, s.x, w.rms_ffn_weight[l], dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            MatrixVectorCollection.matmul(s.hb, s.xb, w.w1[l], dim, p.hidden_dim);
            MatrixVectorCollection.matmul(s.hb2, s.xb, w.w3[l], dim, p.hidden_dim);

            fusedSiluEwiseMul(hidden_dim, s.hb, s.hb2);

            // final matmul to get the output of the ffn
            MatrixVectorCollection.matmul(s.xb, s.hb, w.w2[l], p.hidden_dim, dim);

            residualConnection(s.x, s.xb, dim);
        }

        // final rmsnorm
        rmsnorm(s.x, s.x, w.rms_final_weight, dim);
        // MatrixVectorCollection.matmul(s.logits, s.x, w.wcls, dim, p.vocab_size);
        // MatrixVectorCollection.matmul(s.logits, s.x, w.weightTensor, dim,

        // invoke TornadoVM to run matmul on the GPU
        //MatrixVectorCollection.matrixVectorSimple(s.logits, s.x, w.weightTensor, dim, transformer.config.vocab_size);
        if (Transformer.USE_GPU) {
            // Runs with level Zero on the Intel Integrated GPU
            MatrixVectorCollection.matMulOnGPU(kernel, s.computeBundle, s.logits, s.x, w.weightTensor, dim, transformer.config.vocab_size);

        } else {
            MatrixVectorCollection.matmul(s.logits, s.x, w.weightTensor, dim, transformer.config.vocab_size);
        }

        return s.logits.segment();
    }

    /**
     * Applies the SwiGLU non-linearity to the output of the first layer in the
     * Transformer model.
     *
     * @param hidden_dim
     *            The hidden dimension of the model.
     * @param out
     *            The output array to store the result.
     * @param hb2
     *            The array representing the hidden layer output.
     */
    private static void fusedSiluEwiseMul(int hidden_dim, FloatArray out, FloatArray hb2) {
        for (int i = 0; i < hidden_dim; i++) {
            float val = out.get(i);
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + Math.exp(-val)));
            // elementwise multiply with w3(x)
            out.set(i, (val * hb2.get(i)));
        }
    }

    private static void fusedSiluEwiseMul(int hidden_dim, MemObject out, MemObject hb2) {
        for (int i = 0; i < hidden_dim; i++) {
            float val = out.get(i);
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + Math.exp(-val)));
            // elementwise multiply with w3(x)
            out.set(i, (val * hb2.get(i)));
        }
    }

    /**
     * Applies the residual connection by element-wise addition of the input and
     * residual vectors.
     *
     * @param s
     *            The input vector.
     * @param xb2
     *            The residual vector to be added.
     * @param dim
     *            The dimension of the vectors.
     */
    private static void residualConnection(FloatArray s, FloatArray xb2, int dim) {
        for (int i = 0; i < dim; i++) {
            s.set(i, s.get(i) + xb2.get(i));
        }
    }

    private static void residualConnection(MemObject s, MemObject xb2, int dim) {
        for (int i = 0; i < dim; i++) {
            s.set(i, s.get(i) + xb2.get(i));
        }
    }

    /**
     * Applies root mean square normalization to the input vector.
     *
     * @param o
     *            The output vector.
     * @param x
     *            The input vector.
     * @param weight
     *            The weight values for normalization.
     * @param size
     *            The size of the vectors.
     */
    private static void rmsnorm(FloatArray o, FloatArray x, FloatBuffer weight, int size) {
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x.get(j) * x.get(j);
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / (float) Math.sqrt(ss);
        // normalize and scale
        for (int j = 0; j < size; j++) {
            o.set(j,  weight.get(j) * (ss * x.get(j)));
        }
    }

    private static void rmsnorm(MemObject o, MemObject x, MemObject weight, int size) {
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x.get(j) * x.get(j);
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / (float) Math.sqrt(ss);
        // normalize and scale
        for (int j = 0; j < size; j++) {
            o.set(j,  weight.get(j) * (ss * x.get(j)));
        }
    }

    /**
     * Applies the softmax function to a portion of the input array.
     *
     * @param x
     *            The input array.
     * @param xOffset
     *            The offset within the input array.
     * @param size
     *            The size of the portion to apply softmax.
     */
    static void softmax(float[] x, int xOffset, int size) {
        // find max value (for numerical stability)
        float max_val = x[0 + xOffset];
        for (int i = 1; i < size; i++) {
            if (x[i + xOffset] > max_val) {
                max_val = x[i + xOffset];
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i + xOffset] = (float) Math.exp(x[i + xOffset] - max_val);
            sum += x[i + xOffset];
        }
        // normalize
        for (int i = 0; i < size; i++) {
            x[i + xOffset] /= sum;
        }
    }

    public static float get(MemorySegment segment, int index) {
        return segment.getAtIndex(ValueLayout.JAVA_FLOAT, index);
    }

    static void softmax(MemorySegment x, int xOffset, int size) {
        // find max value (for numerical stability)
        float max_val = get(x, 0 + xOffset);
        for (int i = 1; i < size; i++) {
            if (get(x, i + xOffset) > max_val) {
                max_val = get(x, i + xOffset);
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x.setAtIndex(ValueLayout.JAVA_FLOAT, i + xOffset,  (float) Math.exp(get(x, i + xOffset) - max_val));
            sum += get(x, i + xOffset);
        }
        // normalize
        for (int i = 0; i < size; i++) {
            //x[i + xOffset] /= sum;
            float acc = get(x, i + xOffset);
            x.setAtIndex(ValueLayout.JAVA_FLOAT, i + xOffset, (acc / sum));
        }
    }
}
