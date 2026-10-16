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

kernel void scale(device float* out [[buffer(0)]],
                  device const float* in [[buffer(1)]],
                  device const float* scaleValue [[buffer(2)]],
                  constant MatrixParams& outParams [[buffer(3)]],
                  constant MatrixParams& inParams [[buffer(4)]],
                  constant MatrixParams& scaleParams [[buffer(5)]],
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
    const uint outBase = outParams.offset + outZ * outPlane;
    const uint inBase = inParams.offset + outZ * inPlane;

    const uint outIdx = outBase + (outY * outParams.dimx) + outX;
    const uint inIdx = inBase + (outY * inParams.dimx) + outX;

    out[outIdx] = in[inIdx] * scaleValue[scaleParams.offset];
}
