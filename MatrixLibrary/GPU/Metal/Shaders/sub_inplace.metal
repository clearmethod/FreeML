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

kernel void sub_inplace(device float* out [[buffer(0)]],
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

    const bool outIsRightOperand = (outParams.optionalParams[0] != 0u);
    const uint outIdx = outParams.offset + idx;
    const uint inIdx = inParams.offset + idx;
    if (outIsRightOperand)
    {
        out[outIdx] = in[inIdx] - out[outIdx];
    }
    else
    {
        out[outIdx] = out[outIdx] - in[inIdx];
    }
}
