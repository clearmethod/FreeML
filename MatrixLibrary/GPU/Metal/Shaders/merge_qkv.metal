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

kernel void merge_qkv(device float* packed [[buffer(0)]],
                      device const float* q [[buffer(1)]],
                      device const float* k [[buffer(2)]],
                      device const float* v [[buffer(3)]],
                      constant MatrixParams& packedParams [[buffer(4)]],
                      constant MatrixParams& qParams [[buffer(5)]],
                      constant MatrixParams& kParams [[buffer(6)]],
                      constant MatrixParams& vParams [[buffer(7)]],
                      uint idx [[thread_position_in_grid]])
{
    const uint count = qParams.dimx * qParams.dimy * qParams.dimz;
    if (idx >= count)
    {
        return;
    }

    const uint headDim = qParams.dimx;
    const uint tokens = qParams.dimy;
    const uint nHead = qParams.dimz;
    const uint dC = headDim * nHead;
    const uint headPlane = headDim * tokens;

    const uint head = idx / headPlane;
    const uint rem = idx - head * headPlane;
    const uint row = rem / headDim;
    const uint col = rem - row * headDim;

    const uint srcQ = qParams.offset + idx;
    const uint srcK = kParams.offset + idx;
    const uint srcV = vParams.offset + idx;

    const uint rowPackedBase = packedParams.offset + row * (3u * dC);
    const uint dstQ = rowPackedBase + head * headDim + col;

    packed[dstQ] = q[srcQ];
    packed[dstQ + dC] = k[srcK];
    packed[dstQ + 2u * dC] = v[srcV];
}
