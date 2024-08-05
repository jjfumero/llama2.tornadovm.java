package io.github.mikepapadim;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * This class is used to maintain the state of the model during processing.
 */
public class RunStateFloat extends RunState {
    // current wave of activations
    final FloatArray x; // activation at current time stamp (dim,)
    final FloatArray xb; // same, but inside a residual branch (dim,)
    final FloatArray xb2; // an additional buffer just for convenience (dim,)
    final FloatArray hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    final FloatArray hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    final FloatArray q; // query (dim,)
    final FloatArray k; // key (dim,)
    final FloatArray v; // value (dim,)
    final float[] att; // buffer for scores/attention values (n_heads, seq_len)
    final FloatArray logits; // output logits
    // kv cache
    final float[][] key_cache; // (layer, seq_len, dim)
    final float[][] value_cache; // (layer, seq_len, dim)

    /**
     * Constructs a {@code RunState} object using the provided {@link Config}.
     *
     * @param config
     *            The {@link Config} object containing transformer model
     *            configuration.
     */
    RunStateFloat(Config config) {
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        this.x = new FloatArray(config.dim);
        this.xb = new FloatArray(config.dim);
        this.xb2 = new FloatArray(config.dim);
        this.hb = new FloatArray(config.hidden_dim);
        this.hb2 = new FloatArray(config.hidden_dim);
        this.q = new FloatArray(config.dim);
        this.k = new FloatArray(kv_dim);
        this.v = new FloatArray(kv_dim);
        this.att = new float[config.n_heads * config.seq_len];
        this.logits = new FloatArray(config.vocab_size);
        this.key_cache = new float[config.n_layers][config.seq_len * kv_dim];
        this.value_cache = new float[config.n_layers][config.seq_len * kv_dim];
    }
}