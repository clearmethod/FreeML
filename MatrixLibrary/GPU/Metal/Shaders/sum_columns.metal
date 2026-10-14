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

kernel void sum_columns(device float* out [[buffer(0)]],
                        device const float* in [[buffer(1)]],
                        constant MatrixParams& outParams [[buffer(2)]],
                        constant MatrixParams& inParams [[buffer(3)]],
                        uint col [[thread_position_in_grid]])
{
    if (col >= outParams.dimx || col >= inParams.dimx)
    {
        return;
    }

    float sum = 0.0f;
    for (uint row = 0; row < inParams.dimy; ++row)
    {
        sum += in[inParams.offset + row * inParams.dimx + col];
    }

    out[outParams.offset + col] = sum;
}
