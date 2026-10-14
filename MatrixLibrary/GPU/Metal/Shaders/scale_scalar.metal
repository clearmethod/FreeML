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
    uint optionalParams[16];
};

kernel void scale_scalar(device float* out [[buffer(0)]],
                         device const float* in [[buffer(1)]],
                         constant MatrixParams& outParams [[buffer(2)]],
                         constant MatrixParams& inParams [[buffer(3)]],
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
    const float scaleValue = as_type<float>(outParams.optionalParams[0]);

    out[outIdx] = in[inIdx] * scaleValue;
}
