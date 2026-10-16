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

kernel void scaleadd(device float* out [[buffer(0)]],
                     device const float* in [[buffer(1)]],
                     device const float* scale [[buffer(2)]],
                     device const float* add [[buffer(3)]],
                     constant MatrixParams& outParams [[buffer(4)]],
                     constant MatrixParams& inParams [[buffer(5)]],
                     constant MatrixParams& scaleParams [[buffer(6)]],
                     constant MatrixParams& addParams [[buffer(7)]],
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

    const uint inPlane = inParams.dimx * inParams.dimy;
    const uint addPlane = addParams.dimx * addParams.dimy;

    const uint outBase = outParams.offset + outZ * outPlane;
    const uint inBase = inParams.offset + outZ * inPlane;
    const uint addBase = addParams.offset + outZ * addPlane;

    const uint outIdx = outBase + (outY * outParams.dimx) + outX;
    const uint inIdx = inBase + (outY * inParams.dimx) + outX;
    const uint addIdx = addBase + (outY * addParams.dimx) + outX;

    out[outIdx] = in[inIdx] * scale[scaleParams.offset] + add[addIdx];
}
