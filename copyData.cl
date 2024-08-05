__kernel void copyData(__global float* input, __global float* output) {
	uint idx = get_global_id(0);
	output[idx] = input[idx];
}
