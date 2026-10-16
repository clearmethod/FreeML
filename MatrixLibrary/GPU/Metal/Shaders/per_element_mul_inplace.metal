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

kernel void per_element_mul_inplace(device float* out [[buffer(0)]],
                                    device const float* mask [[buffer(1)]],
                                    constant MatrixParams& outParams [[buffer(2)]],
                                    constant MatrixParams& maskParams [[buffer(3)]],
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

    const uint maskPlane = maskParams.dimx * maskParams.dimy;

    const uint outBase = outParams.offset + outZ * outPlane;
    const uint maskBase = maskParams.offset + outZ * maskPlane;

    const uint outIdx = outBase + (outY * outParams.dimx) + outX;
    const uint maskIdx = maskBase + (outY * maskParams.dimx) + outX;

    out[outIdx] = out[outIdx] * mask[maskIdx];
}
