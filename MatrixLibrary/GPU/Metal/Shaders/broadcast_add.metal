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

kernel void broadcast_add(device float* out [[buffer(0)]],
                          device const float* bias [[buffer(1)]],
                          constant MatrixParams& outParams [[buffer(2)]],
                          constant MatrixParams& biasParams [[buffer(3)]],
                          uint idx [[thread_position_in_grid]])
{
    const uint outPlane = outParams.dimx * outParams.dimy;
    const uint outCount = outPlane * outParams.dimz;
    if (idx >= outCount)
    {
        return;
    }

    const uint outX = idx % outParams.dimx;
    const uint outY = (idx / outParams.dimx) % outParams.dimy;
    const uint outZ = idx / outPlane;

    const uint outBase = outParams.offset + outZ * outPlane;
    const uint outIdx = outBase + (outY * outParams.dimx) + outX;
    const uint biasIdx = biasParams.offset + outX;

    out[outIdx] += bias[biasIdx];
}
