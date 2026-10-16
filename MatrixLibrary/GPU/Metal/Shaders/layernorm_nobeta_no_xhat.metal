#include <metal_stdlib>
using namespace metal;

struct MatrixParams
{
    uint dimx;
    uint dimy;
    uint dimz;
    uint offset;
    uint uniqueId;
    uint pad0;
    uint pad1;
    uint pad2;
};

#define GROUP_SIZE 256

kernel void layernorm_nobeta_no_xhat(device float* out [[buffer(0)]],
                                     device const float* in [[buffer(1)]],
                                     device const float* gamma [[buffer(2)]],
                                     constant MatrixParams& outParams [[buffer(3)]],
                                     constant MatrixParams& inParams [[buffer(4)]],
                                     constant MatrixParams& gammaParams [[buffer(5)]],
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

    const uint inRow = inParams.offset + outZ * inPlane + outY * inParams.dimx;
    const uint outRow = outParams.offset + outZ * outPlane + outY * outParams.dimx;

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

    for (uint i = tid; i < n; i += GROUP_SIZE)
    {
        const float normalized = (in[inRow + i] - mean) * invStd;
        out[outRow + i] = normalized * gamma[gammaParams.offset + i];
    }
}
