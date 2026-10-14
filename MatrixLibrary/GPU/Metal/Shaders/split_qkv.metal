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

kernel void split_qkv(device float* q [[buffer(0)]],
                      device float* k [[buffer(1)]],
                      device float* v [[buffer(2)]],
                      device const float* packed [[buffer(3)]],
                      constant MatrixParams& qParams [[buffer(4)]],
                      constant MatrixParams& kParams [[buffer(5)]],
                      constant MatrixParams& vParams [[buffer(6)]],
                      constant MatrixParams& packedParams [[buffer(7)]],
                      uint idx [[thread_position_in_grid]])
{
    const uint count = qParams.dimx * qParams.dimy * qParams.dimz;
    if (idx >= count)
    {
        return;
    }

    const uint dC = qParams.dimx;
    const uint row = idx / dC;
    const uint col = idx - row * dC;
    const uint packedStride = dC * 3u;
    const uint packedBase = packedParams.offset + row * packedStride;

    const uint qIdx = qParams.offset + idx;
    const uint kIdx = kParams.offset + idx;
    const uint vIdx = vParams.offset + idx;

    q[qIdx] = packed[packedBase + col];
    k[kIdx] = packed[packedBase + col + dC];
    v[vIdx] = packed[packedBase + col + 2u * dC];
}
