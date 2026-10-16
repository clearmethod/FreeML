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

kernel void layernorm_backwards_params(device float* dgamma [[buffer(0)]],
                                       device float* dbeta [[buffer(1)]],
                                       device const float* dy [[buffer(2)]],
                                       device const float* xhat [[buffer(3)]],
                                       constant MatrixParams& dgammaParams [[buffer(4)]],
                                       constant MatrixParams& dbetaParams [[buffer(5)]],
                                       constant MatrixParams& dyParams [[buffer(6)]],
                                       constant MatrixParams& xhatParams [[buffer(7)]],
                                       uint col [[thread_position_in_grid]])
{
    if (col >= dgammaParams.dimx)
    {
        return;
    }

    const uint rows = dyParams.dimy * dyParams.dimz;
    const uint dyPlane = dyParams.dimx * dyParams.dimy;
    const uint xhatPlane = xhatParams.dimx * xhatParams.dimy;

    float sumDy = 0.0f;
    float sumDyXhat = 0.0f;
    for (uint row = 0; row < rows; ++row)
    {
        const uint z = row / dyParams.dimy;
        const uint y = row - z * dyParams.dimy;
        const uint dyBase = dyParams.offset + z * dyPlane + y * dyParams.dimx;
        const uint xhatBase = xhatParams.offset + z * xhatPlane + y * xhatParams.dimx;
        const float dyValue = dy[dyBase + col];
        sumDy += dyValue;
        sumDyXhat += dyValue * xhat[xhatBase + col];
    }

    dbeta[dbetaParams.offset + col] = sumDy;
    dgamma[dgammaParams.offset + col] = sumDyXhat;
}
