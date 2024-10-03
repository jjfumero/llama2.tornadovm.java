package io.github.mikepapadim;

import io.github.mikepapadim.gpu.shared.ComputeBundle;
import io.github.mikepapadim.gpu.shared.MemObject;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/**
 * The Transformer class represents a neural network model with hyperparameters,
 * weights, and state information for performing forward passes.
 */
public class Transformer {

    public static boolean USE_JAVA = Boolean.parseBoolean(System.getProperty("llama2.java", "FALSE"));
    public static boolean USE_LEVEL_ZERO = true;
    public static boolean USE_GPU = true;
    /**
     * The hyperparameters of the architecture (the blueprint).
     */
    Config config;

    /**
     * The weights of the model.
     */
    Weights weights;

    /**
     * Buffers for the "wave" of activations in the forward pass.
     */
    RunState state;

    /**
     * Size of the checkpoint file in bytes.
     */
    private final long fileSize;

    /**
     * Constructs a Transformer by loading the model checkpoint from a file.
     *
     * @param checkpointPath
     *            The path to the checkpoint file.
     * @throws IOException
     *             If an I/O error occurs while reading the checkpoint file.
     */
    public Transformer(String checkpointPath) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(Paths.get(checkpointPath), StandardOpenOption.READ)) {

            this.fileSize = fileChannel.size();
            if (USE_LEVEL_ZERO) {
                ComputeBundle computeBundle = new ComputeBundle();
                // Initialize GPU/Level Zero Platform
                computeBundle.initializeLevelZeroPlatform("kernels.spv");

                MemObject data = computeBundle.allocateSharedWithSegment(this.fileSize);

                // Read the entire file into the Shared Memory Segment
                fileChannel.read(data.segment().asByteBuffer());

                int configSize = 7 * Integer.BYTES;

                // Read in the config header
                MemorySegment configSegment = data.segment().asSlice(0, configSize);
                ByteBuffer configBuffer = configSegment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
                this.config = new Config(configBuffer);
                System.out.println(this.config);

                this.state = new RunStateSharedMem(this.config, computeBundle);

                // Move the position to the beginning of the weights data
                MemObject weightsSegment = computeBundle.allocateSharedWithSegment(computeSize(configSize, data));
                weightsSegment.segment().copyFrom(data.segment().asSlice(configSize));

                this.weights = new WeightsShared(this.config, weightsSegment, computeBundle);

            } else {
                Arena memoryArena = Arena.ofAuto();
                MemorySegment data = memoryArena.allocate(fileSize, 1);

                // Read the entire file into the MemorySegment
                fileChannel.read(data.asByteBuffer());

                int configSize = 7 * Integer.BYTES;

                // Read in the config header
                MemorySegment configSegment = data.asSlice(0, configSize);
                ByteBuffer configBuffer = configSegment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
                this.config = new Config(configBuffer);
                System.out.println(this.config);

                this.state = new RunStateFloat(this.config);

                // Move the position to the beginning of the weights data
                MemorySegment weightsSegment = data.asSlice(configSize);

                this.weights = new WeightsFP32(this.config, weightsSegment);
            }
        }
    }

    private long computeSize(long offset, MemObject data) {
        return (data.segment().byteSize() - offset) / Float.BYTES;
    }
}
