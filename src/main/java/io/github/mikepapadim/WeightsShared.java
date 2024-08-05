package io.github.mikepapadim;

import io.github.mikepapadim.gpu.shared.ComputeBundle;
import io.github.mikepapadim.gpu.shared.MemObject;
import uk.ac.manchester.tornado.api.types.tensors.Shape;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

/**
 * The Weights class represents the weight parameters of a Transformer model,
 * including various weight matrices for token embeddings, attention mechanisms,
 * feedforward networks, and classifier logits.
 */
public class WeightsShared implements Weights {

    // token embedding table
    final MemObject token_embedding_table; // (vocab_size, dim)

    // weights for rmsnorms
    final MemObject[] rms_att_weight; // (layer, dim) rmsnorm weights

    // weights for matmuls. note dim == n_heads * head_size
    final MemObject[] wq; // (layer, dim, n_heads * head_size)
    final MemObject[] wk; // (layer, dim, n_kv_heads * head_size)
    final MemObject[] wv; // (layer, dim, n_kv_heads * head_size)
    final MemObject[] wo; // (layer, n_heads * head_size, dim)

    // weights for ffn
    final MemObject[] rms_ffn_weight; // (layer, dim)
    final MemObject[] w1; // (layer, hidden_dim, dim)
    final MemObject[] w2; // (layer, dim, hidden_dim)
    final MemObject[] w3; // (layer, hidden_dim, dim)

    // final rmsnorm
    final MemObject rms_final_weight; // (dim,)

    final MemObject wcls; // (vocab_size, dim)

    MemObject weightTensor; // vocabInTensor

    ComputeBundle computeBundle;

    /**
     * Constructs Weights by parsing information from a checkpoint's memory segment.
     *
     * @param config
     *            The configuration of the Transformer model.
     * @param memObject
     *            The memory segment containing weight information.
     */
    WeightsShared(Config config, MemObject memObject, ComputeBundle computeBundle) {
        long[] position = new long[] { 0 };
        this.computeBundle = computeBundle;
        FloatBuffer referenceA = takeFloats(memObject.segment(), position, config.vocab_size, config.dim);
        this.token_embedding_table = takeFloats(memObject, position, config.vocab_size, config.dim);
        this.rms_att_weight = takeArray(memObject, position, config.n_layers, config.dim);
        this.wq = takeTensors(memObject, position, config.n_layers, config.dim, config.n_heads * config.head_size);
        this.wk = takeTensors(memObject, position, config.n_layers, config.dim, config.n_kv_heads * config.head_size);
        this.wv = takeTensors(memObject, position, config.n_layers, config.dim, config.n_kv_heads * config.head_size);
        this.wo = takeArray(memObject, position, config.n_layers, config.n_heads * config.head_size, config.dim);
        this.rms_ffn_weight = takeArray(memObject, position, config.n_layers, config.dim);
        this.w1 = takeArray(memObject, position, config.n_layers, config.hidden_dim, config.dim);
        this.w2 = takeArray(memObject, position, config.n_layers, config.dim, config.hidden_dim);
        this.w3 = takeArray(memObject, position, config.n_layers, config.hidden_dim, config.dim);
        this.rms_final_weight = takeFloats(memObject, position, config.dim);
        position[0] += ((long) config.seq_len * config.head_size / 2) * Float.BYTES; // skip what used to be freq_cis_real (for RoPE)
        position[0] += ((long) config.seq_len * config.head_size / 2) * Float.BYTES; // skip what used to be freq_cis_imag (for RoPE)
        FloatBuffer referenceB = config.shared_weights ? referenceA : takeFloats(memObject.segment(), position, config.vocab_size, config.dim);
        this.wcls = config.shared_weights ? this.token_embedding_table : takeFloats(memObject, position, config.vocab_size, config.dim);
        this.weightTensor = getWeightTensor(wcls, referenceB.remaining());
    }

    private FloatBuffer takeFloats(MemorySegment memorySegment, long[] position, int... dims) {
        long totalBytes = 1;
        for (int d : dims) {
            totalBytes *= d;
        }
        totalBytes *= Float.BYTES;
        MemorySegment slice = memorySegment.asSlice(position[0], totalBytes);
        return slice.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    }

    MemObject getWeightTensor(MemObject buffer, int size) {
        //Shape shape = new Shape(size);
        MemObject t = computeBundle.allocateSharedWithSegment(buffer.segment().byteSize() / ValueLayout.JAVA_FLOAT.byteSize());
        t.segment().copyFrom(buffer.segment());
        return t;
    }

    MemObject takeFloats(MemObject memObject, long[] position, int... dims) {
        long totalBytes = 1;
        for (int d : dims) {
            totalBytes *= d;
        }
        totalBytes *= Float.BYTES;
        int sizeSegment = (int) (totalBytes / Float.BYTES);
        MemObject newSharedObject = computeBundle.allocateSharedWithSegment(sizeSegment);
        MemorySegment slice = memObject.segment().asSlice(position[0], totalBytes);
        newSharedObject.segment().copyFrom(slice);
        position[0] += totalBytes;
        FloatBuffer b = newSharedObject.segment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        for (int i = 0;  i < sizeSegment; i++) {
            newSharedObject.set(i, b.get(i));
        }
        return newSharedObject;
    }

    MemObject[] takeArray(MemObject memObject, long[] position, int dim0, int... dims) {
        MemObject[] segments = new MemObject[dim0];
        for (int i = 0; i < dim0; ++i) {
            segments[i] = takeFloats(memObject, position, dims);
        }
        return segments;
    }

    MemObject[] takeTensors(MemObject memObject, long[] position, int dim0, int... dims) {
        MemObject[] weightTensors = new MemObject[dim0];
        for (int i = 0; i < dim0; ++i) {
            weightTensors[i] = takeTensor(memObject, position, dims);
        }
        return weightTensors;
    }

    MemObject takeTensor(MemObject memObject, long[] position, int... dims) {
        long totalBytes = 1;

        for (int d : dims) {
            totalBytes *= d;
        }
        totalBytes *= Float.BYTES;
        MemorySegment slice = memObject.segment().asSlice(position[0], totalBytes);
        position[0] += totalBytes;
        long[] longArray = Arrays.stream(dims).mapToLong(i -> i).toArray();
        Shape shape = new Shape(longArray);
        MemObject t = computeBundle.allocateSharedWithSegment(shape.getSize());
        t.segment().copyFrom(slice);
        return t;
    }
}