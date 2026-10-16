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

kernel void mul_strided_transposeright(device float* out [[buffer(0)]],
                                       device const float* left [[buffer(1)]],
                                       device const float* right [[buffer(2)]],
                                       constant MatrixParams& outParams [[buffer(3)]],
                                       constant MatrixParams& leftParams [[buffer(4)]],
                                       constant MatrixParams& rightParams [[buffer(5)]],
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

    uint outRowStride = outParams.optionalParams[0];
    uint outColStride = outParams.optionalParams[1];
    uint leftRowStride = leftParams.optionalParams[0];
    uint leftColStride = leftParams.optionalParams[1];
    uint rightRowStride = rightParams.optionalParams[0];
    uint rightColStride = rightParams.optionalParams[1];

    if (outRowStride == 0u) outRowStride = outParams.dimx;
    if (outColStride == 0u) outColStride = 1u;
    if (leftRowStride == 0u) leftRowStride = leftParams.dimx;
    if (leftColStride == 0u) leftColStride = 1u;
    if (rightRowStride == 0u) rightRowStride = rightParams.dimx;
    if (rightColStride == 0u) rightColStride = 1u;

    const uint outBase = outParams.offset + (outParams.batch == 1u ? 0u : outZ / outParams.channels) * outParams.optionalParams[3] + (outParams.channels == 1u ? 0u : outZ % outParams.channels) * outParams.optionalParams[2];
    const uint leftBase = leftParams.offset + (leftParams.batch == 1u ? 0u : outZ / outParams.channels) * leftParams.optionalParams[3] + (leftParams.channels == 1u ? 0u : outZ % outParams.channels) * leftParams.optionalParams[2];
    const uint rightBase = rightParams.offset + (rightParams.batch == 1u ? 0u : outZ / outParams.channels) * rightParams.optionalParams[3] + (rightParams.channels == 1u ? 0u : outZ % outParams.channels) * rightParams.optionalParams[2];

    float sum = 0.0f;
    for (uint k = 0; k < leftParams.dimx; ++k)
    {
        const uint leftIdx = leftBase + (outY * leftRowStride) + k * leftColStride;
        const uint rightIdx = rightBase + (outX * rightRowStride) + k * rightColStride;
        sum += left[leftIdx] * right[rightIdx];
    }

    out[outBase + (outY * outRowStride) + outX * outColStride] = sum;
}
