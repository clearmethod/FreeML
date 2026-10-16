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

// All 256 threads participate, including lanes beyond the feature count.
inline float reduce_sum(threadgroup float* scratch, uint tid, float value)
{
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
            scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float result = scratch[0];
    // Every lane must read the result before scratch is reused.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

kernel void layernorm_backwards_dx(device float* dx [[buffer(0)]],
                                   device const float* dy [[buffer(1)]],
                                   device const float* in [[buffer(2)]],
                                   device const float* gamma [[buffer(3)]],
                                   constant MatrixParams& dxParams [[buffer(4)]],
                                   constant MatrixParams& dyParams [[buffer(5)]],
                                   constant MatrixParams& inParams [[buffer(6)]],
                                   constant MatrixParams& gammaParams [[buffer(7)]],
                                   uint3 groupId [[threadgroup_position_in_grid]],
                                   uint3 threadId [[thread_position_in_threadgroup]],
                                   threadgroup float* gsBuf [[threadgroup(0)]],
                                   threadgroup float* gsBuf2 [[threadgroup(1)]],
                                   threadgroup float* gsBuf3 [[threadgroup(2)]])
{
    const uint tid = threadId.x;
    const uint row = groupId.x;

    const uint rows = dxParams.dimy * dxParams.dimz;
    if (row >= rows)
    {
        return;
    }

    const uint outZ = row / dxParams.dimy;
    const uint outY = row - outZ * dxParams.dimy;
    const uint n = dxParams.dimx;

    const uint dxPlane = dxParams.dimx * dxParams.dimy;
    const uint dyPlane = dyParams.dimx * dyParams.dimy;
    const uint inPlane = inParams.dimx * inParams.dimy;

    const uint dxRow = dxParams.offset + outZ * dxPlane + outY * dxParams.dimx;
    const uint dyRow = dyParams.offset + outZ * dyPlane + outY * dyParams.dimx;
    const uint inRow = inParams.offset + outZ * inPlane + outY * inParams.dimx;

    float sum = 0.0f;
    for (uint i = tid; i < n; i += GROUP_SIZE)
        sum += in[inRow + i];
    const float mean = reduce_sum(gsBuf, tid, sum) / float(n);

    float var = 0.0f;
    for (uint i = tid; i < n; i += GROUP_SIZE)
    {
        const float diff = in[inRow + i] - mean;
        var += diff * diff;
    }
    const float invStd = rsqrt(reduce_sum(gsBuf, tid, var) / float(n) + 1e-5f);
    const float invStd3 = invStd * invStd * invStd;
    const float invN = 1.0f / float(n);

    float partialVar = 0.0f, partialMean = 0.0f, partialDiff = 0.0f;
    for (uint i = tid; i < n; i += GROUP_SIZE)
    {
        const float diff = in[inRow + i] - mean;
        const float dXhat = dy[dyRow + i] * gamma[gammaParams.offset + i];
        partialVar += dXhat * diff * (-0.5f) * invStd3;
        partialMean += dXhat * (-invStd);
        partialDiff += diff;
    }
    gsBuf[tid] = partialVar;
    gsBuf2[tid] = partialMean;
    gsBuf3[tid] = partialDiff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            gsBuf[tid] += gsBuf[tid + stride];
            gsBuf2[tid] += gsBuf2[tid + stride];
            gsBuf3[tid] += gsBuf3[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float dVar = gsBuf[0];
    const float dMean = gsBuf2[0] + dVar * (-2.0f) * gsBuf3[0] * invN;

    for (uint i = tid; i < n; i += GROUP_SIZE)
    {
        const float diff = in[inRow + i] - mean;
        const float dXhat = dy[dyRow + i] * gamma[gammaParams.offset + i];
        dx[dxRow + i] = dXhat * invStd
                      + dVar * 2.0f * diff * invN
                      + dMean * invN;
    }
}
