package io.github.mikepapadim;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.stream.IntStream;

import io.github.mikepapadim.gpu.shared.ComputeBundle;
import io.github.mikepapadim.gpu.shared.MemObject;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.tensors.TensorFP32;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroKernel;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeGroupDispatch;

/**
 * This class contains a set of Matrix and Vector multiplication methods
 * implemented using TornadoVM's API.
 */
public class MatrixVectorCollection {

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input
     * vector (x). It uses Graal's auto-vectorization for the matrix multiplication.
     * However, if the {@link Llama2} flag is set to true, it
     * employs explicit vectorization using the Vector API for further optimization.
     *
     * @param xout
     *            The output vector of the matrix multiplication.
     * @param x
     *            The input vector to be multiplied with the weight matrix.
     * @param w
     *            The weight matrix represented as a {@link FloatBuffer}.
     * @param n
     *            The number of columns in the weight matrix and the size of the
     *            input vector.
     * @param d
     *            The number of rows in the weight matrix and the size of the output
     *            vector.
     */
    static void matmul(float[] xout, float[] x, FloatBuffer w, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        MemorySegment wSegment = MemorySegment.ofBuffer(w);
        IntStream.range(0, d).parallel().forEach(i -> {
            float val = 0f;
            int j = 0;
            if (Llama2.USE_VECTOR_API) {
                VectorSpecies<Float> species = FloatVector.SPECIES_256;
                FloatVector sum0 = FloatVector.zero(species);
                FloatVector sum1 = FloatVector.zero(species);
                FloatVector sum2 = FloatVector.zero(species);
                FloatVector sum3 = FloatVector.zero(species);
                int width = species.length();
                int upperBound = n - n % (4 * width);
                for (; j < upperBound; j += 4 * width) {
                    var wj0 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 0 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var wj1 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 1 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var wj2 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 2 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var wj3 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 3 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var xj0 = FloatVector.fromArray(species, x, j + 0 * width);
                    var xj1 = FloatVector.fromArray(species, x, j + 1 * width);
                    var xj2 = FloatVector.fromArray(species, x, j + 2 * width);
                    var xj3 = FloatVector.fromArray(species, x, j + 3 * width);
                    sum0 = wj0.fma(xj0, sum0);
                    sum1 = wj1.fma(xj1, sum1);
                    sum2 = wj2.fma(xj2, sum2);
                    sum3 = wj3.fma(xj3, sum3);
                }
                val = sum0.add(sum1).add(sum2).add(sum3).reduceLanes(VectorOperators.ADD);
            }

            // Graal's auto-vectorization.
            int upperBound = n & ~3;
            float[] sum = new float[4];
            for (; j < upperBound; j += sum.length) {
                sum[0] += w.get(i * n + j + 0) * x[j + 0];
                sum[1] += w.get(i * n + j + 1) * x[j + 1];
                sum[2] += w.get(i * n + j + 2) * x[j + 2];
                sum[3] += w.get(i * n + j + 3) * x[j + 3];
            }
            val += sum[0] + sum[1] + sum[2] + sum[3];

            for (; j < n; j++) {
                val += w.get(i * n + j) * x[j];
            }
            xout[i] = val;
        });
    }

    static void matmul(FloatArray xout, FloatArray x, FloatBuffer w, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        MemorySegment wSegment = MemorySegment.ofBuffer(w);
        IntStream.range(0, d).parallel().forEach(i -> {
            float val = 0f;
            int j = 0;
            // Graal's auto-vectorization.
            int upperBound = n & ~3;
            float[] sum = new float[4];
            for (; j < upperBound; j += sum.length) {
                sum[0] += w.get(i * n + j + 0) * x.get(j + 0);
                sum[1] += w.get(i * n + j + 1) * x.get(j + 1);
                sum[2] += w.get(i * n + j + 2) * x.get(j + 2);
                sum[3] += w.get(i * n + j + 3) * x.get(j + 3);
            }
            val += sum[0] + sum[1] + sum[2] + sum[3];

            for (; j < n; j++) {
                val += w.get(i * n + j) * x.get(j);
            }
            xout.set(i, val);
        });
    }

    static void matmul(float[] xout, float[] x, TensorFP32 weightTensor, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        MemorySegment wSegment = weightTensor.getSegment();
        IntStream.range(0, d).parallel().forEach(i -> {
            float val = 0f;
            int j = 0;
            if (Llama2.USE_VECTOR_API) {
                VectorSpecies<Float> species = FloatVector.SPECIES_256;
                FloatVector sum0 = FloatVector.zero(species);
                FloatVector sum1 = FloatVector.zero(species);
                FloatVector sum2 = FloatVector.zero(species);
                FloatVector sum3 = FloatVector.zero(species);
                int width = species.length();
                int upperBound = n - n % (4 * width);
                for (; j < upperBound; j += 4 * width) {
                    var wj0 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 0 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var wj1 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 1 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var wj2 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 2 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var wj3 = FloatVector.fromMemorySegment(species, wSegment, (i * n + j + 3 * width) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                    var xj0 = FloatVector.fromArray(species, x, j + 0 * width);
                    var xj1 = FloatVector.fromArray(species, x, j + 1 * width);
                    var xj2 = FloatVector.fromArray(species, x, j + 2 * width);
                    var xj3 = FloatVector.fromArray(species, x, j + 3 * width);
                    sum0 = wj0.fma(xj0, sum0);
                    sum1 = wj1.fma(xj1, sum1);
                    sum2 = wj2.fma(xj2, sum2);
                    sum3 = wj3.fma(xj3, sum3);
                }
                val = sum0.add(sum1).add(sum2).add(sum3).reduceLanes(VectorOperators.ADD);
            }

            // Graal's auto-vectorization.
            int upperBound = n & ~3;
            float[] sum = new float[4];
            for (; j < upperBound; j += sum.length) {
                sum[0] += weightTensor.get(i * n + j + 0) * x[j + 0];
                sum[1] += weightTensor.get(i * n + j + 1) * x[j + 1];
                sum[2] += weightTensor.get(i * n + j + 2) * x[j + 2];
                sum[3] += weightTensor.get(i * n + j + 3) * x[j + 3];
            }
            val += sum[0] + sum[1] + sum[2] + sum[3];

            for (; j < n; j++) {
                val += weightTensor.get(i * n + j) * x[j];
            }
            xout[i] = val;
        });
    }

    static void matmul(FloatArray xout, FloatArray x, TensorFP32 weightTensor, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        IntStream.range(0, d).parallel().forEach(i -> {
            float val = 0f;
            int j = 0;

            // Graal's auto-vectorization.
            int upperBound = n & ~3;
            float[] sum = new float[4];
            for (; j < upperBound; j += sum.length) {
                sum[0] += weightTensor.get(i * n + j + 0) * x.get(j + 0);
                sum[1] += weightTensor.get(i * n + j + 1) * x.get(j + 1);
                sum[2] += weightTensor.get(i * n + j + 2) * x.get(j + 2);
                sum[3] += weightTensor.get(i * n + j + 3) * x.get(j + 3);
            }
            val += sum[0] + sum[1] + sum[2] + sum[3];

            for (; j < n; j++) {
                val += weightTensor.get(i * n + j) * x.get(j);
            }
            xout.set(i, val);
        });
    }

    static void matmul(MemObject xout, MemObject x, MemObject w, int n, int d) {
        IntStream.range(0, d).parallel().forEach(i -> {
            float val = 0f;
            int j = 0;
            // Graal's auto-vectorization.
            int upperBound = n & ~3;
            float[] sum = new float[4];
            for (; j < upperBound; j += sum.length) {
                sum[0] += w.get(i * n + j + 0) * x.get(j + 0);
                sum[1] += w.get(i * n + j + 1) * x.get(j + 1);
                sum[2] += w.get(i * n + j + 2) * x.get(j + 2);
                sum[3] += w.get(i * n + j + 3) * x.get(j + 3);
            }
            val += sum[0] + sum[1] + sum[2] + sum[3];

            for (; j < n; j++) {
                val += w.get(i * n + j) * x.get(j);
            }
            xout.set(i, val);
        });
    }

    static void matMulOnGPU(LevelZeroKernel kernel, ComputeBundle computeBundle, MemObject xout, MemObject x, MemObject w, int n, int numThreads) {
        if (computeBundle.getMatMulDispatcher() == null) {
            ComputeBundle.DispacherMeta dispacherMeta = computeBundle.runMatMul(kernel, xout.buffer(), x.buffer(), w.buffer(), n, numThreads);
            computeBundle.setDispatchMatMul(dispacherMeta);
        } else {
            ComputeBundle.DispacherMeta dispatcher = computeBundle.getMatMulDispatcher();
            computeBundle.dispatchMatMul(kernel, dispatcher);
        }
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input
     * vector (x). Is uses parallel streams for optimization.
     *
     * @param xout
     *            The output vector of the matrix multiplication.
     * @param x
     *            The input vector to be multiplied with the weight matrix.
     * @param w
     *            The weight matrix represented as a {@link FloatBuffer}.
     * @param n
     *            The number of columns in the weight matrix and the size of the
     *            input vector.
     * @param d
     *            The number of rows in the weight matrix and the size of the output
     *            vector.
     */
    static void matrixVectorMultiply(float[] xout, float[] x, FloatBuffer w, int n, int d) {
        IntStream.range(0, d).parallel().forEach(i -> {
            float val = 0f;
            for (int j = 0; j < n; j++) {
                val += w.get(i * n + j) * x[j];
            }
            xout[i] = val;
        });
    }

    /**
     * Performs matrix multiplication between the weight matrix (W) and the input
     * vector (x). It uses the @Parallel annotation of TornadoVM to signify to the
     * compiler that the loop is data parallel. The weight matrix w is represented
     * by a TornadoVM {@link FloatArray}, which resides off-heap.
     *
     * @param xout
     *            The output vector of the matrix multiplication.
     * @param x
     *            The input vector to be multiplied with the weight matrix.
     * @param w
     *            The weight matrix represented as a {@link FloatArray}.
     * @param n
     *            The number of columns in the weight matrix and the size of the
     *            input vector.
     * @param d
     *            The number of rows in the weight matrix and the size of the output
     *            vector.
     */
    static void matrixVectorSimple(float[] xout, float[] x, FloatArray w, int n, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            float val = 0f;
            for (int j = 0; j < n; j++) {
                val += w.get(i * n + j) * x[j];
            }
            xout[i] = val;
        }
    }

    static void matrixVectorSimple(float[] xout, float[] x, TensorFP32 w, int n, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            float val = 0f;
            for (int j = 0; j < n; j++) {
                val += w.get(i * n + j) * x[j];
            }
            xout[i] = val;
        }
    }

    static void matrixVectorSimple(FloatArray xout, FloatArray x, TensorFP32 w, int n, int d) {
        for (@Parallel int i = 0; i < d; i++) {
            float val = 0f;
            for (int j = 0; j < n; j++) {
                val += w.get(i * n + j) * x.get(j);
            }
            xout.set(i, val);
        }
    }

}
