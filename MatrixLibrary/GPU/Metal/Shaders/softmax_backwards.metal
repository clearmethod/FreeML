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

kernel void softmax_backwards(device float* out [[buffer(0)]],
                              device const float* soft [[buffer(1)]],
                              device const float* grad [[buffer(2)]],
                              constant MatrixParams& outParams [[buffer(3)]],
                              constant MatrixParams& softParams [[buffer(4)]],
                              constant MatrixParams& gradParams [[buffer(5)]],
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
    const uint softPlane = softParams.dimx * softParams.dimy;
    const uint gradPlane = gradParams.dimx * gradParams.dimy;

    const uint outRow = outParams.offset + outZ * outPlane + outY * outParams.dimx;
    const uint softRow = softParams.offset + outZ * softPlane + outY * softParams.dimx;
    const uint gradRow = gradParams.offset + outZ * gradPlane + outY * gradParams.dimx;

    float partialDot = 0.0f;
    for (uint i = tid; i < n; i += GROUP_SIZE)
        partialDot += grad[gradRow + i] * soft[softRow + i];
    gsBuf[tid] = partialDot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
            gsBuf[tid] += gsBuf[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float dot = gsBuf[0];

    for (uint i = tid; i < n; i += GROUP_SIZE)
    {
        const float s = soft[softRow + i];
        const float g = grad[gradRow + i];
        out[outRow + i] = s * (g - dot);
    }
}
