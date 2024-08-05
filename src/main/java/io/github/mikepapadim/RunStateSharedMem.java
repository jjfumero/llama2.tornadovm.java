package io.github.mikepapadim;

import io.github.mikepapadim.gpu.shared.ComputeBundle;
import io.github.mikepapadim.gpu.shared.MemObject;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroKernel;

/**
 * This class is used to maintain the state of the model during processing.
 */
public class RunStateSharedMem extends  RunState {
    // current wave of activations
    final MemObject x; // activation at current time stamp (dim,)
    final MemObject xb; // same, but inside a residual branch (dim,)
    final MemObject xb2; // an additional buffer just for convenience (dim,)
    final MemObject hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    final MemObject hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    final MemObject q; // query (dim,)
    final MemObject k; // key (dim,)
    final MemObject v; // value (dim,)
    final float[] att; // buffer for scores/attention values (n_heads, seq_len)
    final MemObject logits; // output logits
    // kv cache
    final float[][] key_cache; // (layer, seq_len, dim)
    final float[][] value_cache; // (layer, seq_len, dim)

    ComputeBundle computeBundle;
    LevelZeroKernel matMulkernel;

    /**
     * Constructs a {@code RunState} object using the provided {@link Config}.
     *
     * @param config
     *            The {@link Config} object containing transformer model
     *            configuration.
     */
    RunStateSharedMem(Config config, ComputeBundle computeBundle) {
        this.computeBundle = computeBundle;
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        this.x = allocate(config.dim);
        this.xb = allocate(config.dim);
        this.xb2 = allocate(config.dim);
        this.hb = allocate(config.hidden_dim);
        this.hb2 = allocate(config.hidden_dim);
        this.q = allocate(config.dim);
        this.k = allocate(kv_dim);
        this.v = allocate(kv_dim);
        this.att = new float[(config.n_heads * config.seq_len)];
        this.logits = allocate(config.vocab_size);
        this.key_cache = new float[config.n_layers][config.seq_len * kv_dim];
        this.value_cache = new float[config.n_layers][config.seq_len * kv_dim];
        matMulkernel = computeBundle.createKernel("matMul");
    }

    private MemObject allocate(long size) {
        return computeBundle.allocateSharedWithSegment(size);
    }
}