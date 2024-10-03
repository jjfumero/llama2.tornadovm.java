// ===============================================================
// Llama2 core OpenCL Kernels
// ===============================================================

// ===============================================================
// Kernels rmsNorm
// ===============================================================
__kernel void rmsnormReduction(__global float *partialSums, __global float *x, __local float* localSums) {
    int idx = get_global_id(0);
    int localIdx = get_local_id(0);
    int groupSize = get_local_size(0);
    int groupID = get_group_id(0);
    localSums[localIdx] = x[idx];

    localSums[localIdx] = localSums[localIdx] * localSums[localIdx];

    for (int stride = groupSize / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localIdx < stride) {
            localSums[localIdx] += localSums[localIdx + stride];
        }
    }
    if (localIdx == 0) {
        partialSums[groupID] = localSums[0];
    }
}

__kernel void rmsnormNormalization(__global float *output, __global float *x, __global float *weight, const float ss) {
    uint idx = get_global_id(0);
	output[idx] = weight[idx] * (ss * x[idx]);
}

// ===============================================================
// Kernels: softmax
// ===============================================================
__kernel void softMaxReduction(__global float *partialMax, __global float *x, __local float* locals) {
    uint idx = get_global_id(0);
    uint localIdx = get_local_id(0);
    uint groupSize = get_local_size(0);
    locals[localIdx] = x[idx];

    for (int stride = groupSize / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localIdx < stride) {
            if (locals[localIdx] < locals[localIdx + stride]) {
                locals[localIdx] = locals[localIdx + stride];
            }
        }
    }
    if (localIdx == 0) {
        partialMax[get_group_id(0)] = locals[0];
    }
}

__kernel void softMaxExpAndSum(__global float *partialSums, __global float *x, __local float* locals, const float maxValue) {
    uint idx = get_global_id(0);
    uint localIdx = get_local_id(0);
    uint groupSize = get_local_size(0);

    locals[localIdx] = exp(locals[localIdx] - maxValue);

    for (int stride = groupSize / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localIdx < stride) {
            locals[localIdx] += locals[localIdx + stride];
        }
    }

    if (localIdx == 0) {
        partialSums[get_group_id(0)] = locals[0];
    }
}

__kernel void softMaxNormalization(__global float *x, const float sum) {
    uint idx = get_global_id(0);
	x[idx] = x[idx] / sum;
}

// ===============================================================
// Kernels: matMul
// ===============================================================
__kernel void matMul(__global float *xout, __global const float *x,  __global const float *w, const int n) {
    uint idx = get_global_id(0);
    float val = 0;
    #pragma unroll 8
    for (int j = 0; j < n; j++) {
        // val = fma(w[idx * n + j], x[j], val);
        val += w[idx * n + j] * x[j];
    }
	xout[idx] = val;
}

// Second approach using OpenCL local memory
__kernel void matMulLocal(__global float *xout,
                           __global const float *x,
                           __global const float* w,
                           __local float *lvector,
                           const int n) {
    // Get the row index for this thread
    int idx = get_global_id(0);

    // Load vector into local memory to reduce global memory access
    int localIdx = get_local_id(0);
    int localSize = get_local_size(0);

    // Load parts of the vector into local memory in chunks
    for (int i = localIdx; i < n; i += localSize) {
        lvector[i] = x[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute the matrix-vector multiplication
    float acc = 0.0f;
    for (int j = 0; j < n; j++) {
        acc += w[idx * n + j] * lvector[j];
    }

    // store final result
    xout[idx] = acc;
}
