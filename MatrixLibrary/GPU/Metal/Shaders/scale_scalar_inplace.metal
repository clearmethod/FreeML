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
    uint optionalParams[16];
};

kernel void scale_scalar_inplace(device float* out [[buffer(0)]],
                                 constant MatrixParams& outParams [[buffer(1)]],
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
    const uint outIdx = outParams.offset + outZ * outPlane + (outY * outParams.dimx) + outX;
    const float scaleValue = as_type<float>(outParams.optionalParams[0]);

    out[outIdx] *= scaleValue;
}
