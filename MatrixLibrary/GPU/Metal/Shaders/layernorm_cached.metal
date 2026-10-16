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

kernel void layernorm_cached(device float* out [[buffer(0)]],
                      device float* xhat [[buffer(1)]],
                      device const float* in [[buffer(2)]],
                      device const float* gamma [[buffer(3)]],
                      device const float* beta [[buffer(4)]],
                      constant MatrixParams& outParams [[buffer(5)]],
                      constant MatrixParams& inParams [[buffer(6)]],
                      constant MatrixParams& gammaParams [[buffer(7)]],
                      constant MatrixParams& betaParams [[buffer(8)]],
                      constant MatrixParams& xhatParams [[buffer(9)]],
                      device float* invStdCache [[buffer(10)]],
                      constant MatrixParams& invStdParams [[buffer(11)]],
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
    const uint n = inParams.dimx;

    const uint inPlane = inParams.dimx * inParams.dimy;
    const uint outPlane = outParams.dimx * outParams.dimy;
    const uint xhatPlane = xhatParams.dimx * xhatParams.dimy;

    const uint inRow = inParams.offset + outZ * inPlane + outY * inParams.dimx;
    const uint outRow = outParams.offset + outZ * outPlane + outY * outParams.dimx;
    const uint xhatRow = xhatParams.offset + outZ * xhatPlane + outY * xhatParams.dimx;

    if (tid == 0)
    {
        float sum = 0.0f;
        for (uint i = 0; i < n; ++i)
        {
            sum += in[inRow + i];
        }
        gsBuf[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = gsBuf[0] / float(n);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0)
    {
        float var = 0.0f;
        for (uint i = 0; i < n; ++i)
        {
            const float diff = in[inRow + i] - mean;
            var += diff * diff;
        }
        gsBuf[0] = var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invStd = rsqrt(gsBuf[0] / float(n) + 1e-5f);

    if (tid == 0) invStdCache[invStdParams.offset + row] = invStd;

    for (uint i = tid; i < n; i += GROUP_SIZE)
    {
        const float normalized = (in[inRow + i] - mean) * invStd;
        xhat[xhatRow + i] = normalized;
        out[outRow + i] = normalized * gamma[gammaParams.offset + i] + beta[betaParams.offset + i];
    }
}
