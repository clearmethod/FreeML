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

kernel void add(device float* out [[buffer(0)]],
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

    const uint leftPlane = leftParams.dimx * leftParams.dimy;
    const uint rightPlane = rightParams.dimx * rightParams.dimy;

    const uint outBase = outParams.offset + outZ * outPlane;
    const uint leftBase = leftParams.offset + outZ * leftPlane;
    const uint rightBase = rightParams.offset + outZ * rightPlane;

    const uint outIdx = outBase + (outY * outParams.dimx) + outX;
    const uint leftIdx = leftBase + (outY * leftParams.dimx) + outX;
    const uint rightIdx = rightBase + (outY * rightParams.dimx) + outX;

    out[outIdx] = left[leftIdx] + right[rightIdx];
}
