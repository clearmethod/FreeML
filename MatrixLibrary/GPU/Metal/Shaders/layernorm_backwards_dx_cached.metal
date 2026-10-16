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

kernel void layernorm_backwards_dx_cached(device float* dx [[buffer(0)]],
                                   device const float* dy [[buffer(1)]],
                                   device const float* xhat [[buffer(2)]],
                                   device const float* gamma [[buffer(3)]],
                                   constant MatrixParams& dxParams [[buffer(4)]],
                                   constant MatrixParams& dyParams [[buffer(5)]],
                                   constant MatrixParams& xhatParams [[buffer(6)]],
                                   constant MatrixParams& gammaParams [[buffer(7)]],
                                   device const float* invStdCache [[buffer(8)]],
                                   constant MatrixParams& invStdParams [[buffer(9)]],
                                   uint3 groupId [[threadgroup_position_in_grid]],
                                   uint3 threadId [[thread_position_in_threadgroup]],
                                   threadgroup float* gsBuf [[threadgroup(0)]],
                                   threadgroup float* gsBuf2 [[threadgroup(1)]])
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
    const uint xhatPlane = xhatParams.dimx * xhatParams.dimy;
    const uint dxRow = dxParams.offset + outZ * dxPlane + outY * dxParams.dimx;
    const uint dyRow = dyParams.offset + outZ * dyPlane + outY * dyParams.dimx;
    const uint xhatRow = xhatParams.offset + outZ * xhatPlane + outY * xhatParams.dimx;
    const float invStd = invStdCache[invStdParams.offset + row];
    const float invN = 1.0f / float(n);

    // Reuse forward statistics and normalized inputs; only gradient sums remain.
    float sumGrad = 0.0f, sumGradHat = 0.0f;
    for (uint i = tid; i < n; i += GROUP_SIZE)
    {
        const float grad = dy[dyRow + i] * gamma[gammaParams.offset + i];
        sumGrad += grad;
        sumGradHat += grad * xhat[xhatRow + i];
    }
    gsBuf[tid] = sumGrad;
    gsBuf2[tid] = sumGradHat;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            gsBuf[tid] += gsBuf[tid + stride];
            gsBuf2[tid] += gsBuf2[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float meanGrad = gsBuf[0] * invN;
    const float meanGradHat = gsBuf2[0] * invN;
    for (uint i = tid; i < n; i += GROUP_SIZE)
        dx[dxRow + i] = invStd * (dy[dyRow + i] * gamma[gammaParams.offset + i]
                                 - meanGrad - xhat[xhatRow + i] * meanGradHat);
}
