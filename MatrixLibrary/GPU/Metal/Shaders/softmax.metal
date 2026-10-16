#include <metal_stdlib>
using namespace metal;

struct MatrixParams
{
    uint dimx;
    uint dimy;
    uint dimz;
    uint offset;
    uint uniqueId;
    uint channels;
    uint batch;
    uint reserved;
};

#define GROUP_SIZE 256

kernel void softmax(device float* out [[buffer(0)]],
                    device const float* in [[buffer(1)]],
                    constant MatrixParams& outParams [[buffer(2)]],
                    constant MatrixParams& inParams [[buffer(3)]],
                    uint3 groupId [[threadgroup_position_in_grid]],
                    uint3 threadId [[thread_position_in_threadgroup]],
                    threadgroup float* gsBuf [[threadgroup(0)]])
{
    const uint tid = threadId.x;
    const uint row = groupId.x;

    const uint rows = outParams.dimy * outParams.dimz;
    if (row >= rows)
    {
        return;
    }

    const uint outZ = row / outParams.dimy;
    const uint outY = row - outZ * outParams.dimy;
    const uint n = outParams.dimx;

    const uint outPlane = outParams.dimx * outParams.dimy;
    const uint inPlane = inParams.dimx * inParams.dimy;

    const uint outRow = outParams.offset + outZ * outPlane + outY * outParams.dimx;
    const uint inRow = inParams.offset + outZ * inPlane + outY * inParams.dimx;

    float localMax = -1e38f;
    for (uint i = tid; i < n; i += GROUP_SIZE)
    {
        const float value = in[inRow + i];
        localMax = value > localMax ? value : localMax;
    }
    gsBuf[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GROUP_SIZE / 2u; stride > 0u; stride >>= 1u)
    {
        if (tid < stride)
        {
            const float other = gsBuf[tid + stride];
            gsBuf[tid] = gsBuf[tid] > other ? gsBuf[tid] : other;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float maxValue = gsBuf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0)
    {
        float sum = 0.0f;
        for (uint i = 0; i < n; ++i)
        {
            sum += exp(in[inRow + i] - maxValue);
        }
        gsBuf[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invSum = 1.0f / gsBuf[0];

    for (uint i = tid; i < n; i += GROUP_SIZE)
    {
        out[outRow + i] = exp(in[inRow + i] - maxValue) * invSum;
    }
}
