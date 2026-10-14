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

kernel void layernorm_backwards_params_nobeta(device float* dgamma [[buffer(0)]],
                                              device const float* dy [[buffer(1)]],
                                              device const float* xhat [[buffer(2)]],
                                              constant MatrixParams& dgammaParams [[buffer(3)]],
                                              constant MatrixParams& dyParams [[buffer(4)]],
                                              constant MatrixParams& xhatParams [[buffer(5)]],
                                              uint col [[thread_position_in_grid]])
{
    if (col >= dgammaParams.dimx)
    {
        return;
    }

    const uint rows = dyParams.dimy * dyParams.dimz;
    const uint dyPlane = dyParams.dimx * dyParams.dimy;
    const uint xhatPlane = xhatParams.dimx * xhatParams.dimy;

    float sumDyXhat = 0.0f;
    for (uint row = 0; row < rows; ++row)
    {
        const uint z = row / dyParams.dimy;
        const uint y = row - z * dyParams.dimy;
        const uint dyBase = dyParams.offset + z * dyPlane + y * dyParams.dimx;
        const uint xhatBase = xhatParams.offset + z * xhatPlane + y * xhatParams.dimx;
        const float dyValue = dy[dyBase + col];
        sumDyXhat += dyValue * xhat[xhatBase + col];
    }

    dgamma[dgammaParams.offset + col] = sumDyXhat;
}
