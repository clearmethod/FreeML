cbuffer MatPacked_Params : register(b0)
{
    uint packed_dimx;
    uint packed_dimy;
    uint packed_dimz;
    uint packed_offset;
    uint packed_uniqueId;
    uint packed_channels;
    uint packed_batch;
    uint packed_reserved;
    uint4 packed_optionalParams[4];
};

cbuffer MatQ_Params : register(b1)
{
    uint q_dimx;
    uint q_dimy;
    uint q_dimz;
    uint q_offset;
    uint q_uniqueId;
    uint q_channels;
    uint q_batch;
    uint q_reserved;
    uint4 q_optionalParams[4];
};

cbuffer MatK_Params : register(b2)
{
    uint k_dimx;
    uint k_dimy;
    uint k_dimz;
    uint k_offset;
    uint k_uniqueId;
    uint k_channels;
    uint k_batch;
    uint k_reserved;
    uint4 k_optionalParams[4];
};

cbuffer MatV_Params : register(b3)
{
    uint v_dimx;
    uint v_dimy;
    uint v_dimz;
    uint v_offset;
    uint v_uniqueId;
    uint v_channels;
    uint v_batch;
    uint v_reserved;
    uint4 v_optionalParams[4];
};

RWStructuredBuffer<float> MatPacked : register(u0);
StructuredBuffer<float> MatQ : register(t1);
StructuredBuffer<float> MatK : register(t2);
StructuredBuffer<float> MatV : register(t3);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    const uint count = q_dimx * q_dimy * q_dimz;
    if (idx >= count)
    {
        return;
    }

    const uint headDim = q_dimx;
    const uint tokens = q_dimy;
    const uint nHead = q_channels;
    const uint dC = headDim * nHead;
    const uint headPlane = headDim * tokens;

    const uint plane = idx / headPlane;
    const uint head = plane % nHead, batch = plane / nHead;
    const uint rem = idx - plane * headPlane;
    const uint row = rem / headDim;
    const uint col = rem - row * headDim;

    const uint srcQ = q_offset + idx;
    const uint srcK = k_offset + idx;
    const uint srcV = v_offset + idx;

    const uint rowPackedBase = packed_offset + (batch * tokens + row) * (3u * dC);
    const uint dstQ = rowPackedBase + head * headDim + col;

    MatPacked[dstQ] = MatQ[srcQ];
    MatPacked[dstQ + dC] = MatK[srcK];
    MatPacked[dstQ + 2u * dC] = MatV[srcV];
}
