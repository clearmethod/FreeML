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

kernel void scaleadd_inplace(device float* out [[buffer(0)]],
                             device const float* scale [[buffer(1)]],
                             device const float* add [[buffer(2)]],
                             constant MatrixParams& outParams [[buffer(3)]],
                             constant MatrixParams& scaleParams [[buffer(4)]],
                             constant MatrixParams& addParams [[buffer(5)]],
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

    const uint addPlane = addParams.dimx * addParams.dimy;
    const uint outBase = outParams.offset + outZ * outPlane;
    const uint addBase = addParams.offset + outZ * addPlane;

    const uint outIdx = outBase + (outY * outParams.dimx) + outX;
    const uint addIdx = addBase + (outY * addParams.dimx) + outX;

    out[outIdx] = out[outIdx] * scale[scaleParams.offset] + add[addIdx];
}
