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

static inline float erf_approx(float x)
{
    float t = 1.0f / (1.0f + 0.5f * fabs(x));
    float tau = t * exp(-x * x - 1.26551223f
                        + t * (1.00002368f
                        + t * (0.37409196f
                        + t * (0.09678418f
                        + t * (-0.18628806f
                        + t * (0.27886807f
                        + t * (-1.13520398f
                        + t * (1.48851587f
                        + t * (-0.82215223f
                        + t * 0.17087277f)))))))));
    return (x >= 0.0f) ? (1.0f - tau) : (tau - 1.0f);
}

kernel void gelu_inplace(device float* out [[buffer(0)]],
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

    const float x = out[outIdx];
    const float erfTerm = erf_approx(x * 0.70710678118f);
    out[outIdx] = 0.5f * x * (1.0f + erfTerm);
}
