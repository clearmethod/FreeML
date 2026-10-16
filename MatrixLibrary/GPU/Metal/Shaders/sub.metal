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

kernel void sub(device float* out [[buffer(0)]],
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

    out[outParams.offset + idx] = left[leftParams.offset + idx] - right[rightParams.offset + idx];
}
