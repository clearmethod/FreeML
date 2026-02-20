cbuffer MatQ_Params : register(b0)
{
    uint q_dimx;
    uint q_dimy;
    uint q_dimz;
    uint q_offset;
    uint q_uniqueId;
    uint q_pad0;
    uint q_pad1;
    uint q_pad2;
    uint4 q_optionalParams[4];
};

cbuffer MatK_Params : register(b1)
{
    uint k_dimx;
    uint k_dimy;
    uint k_dimz;
    uint k_offset;
    uint k_uniqueId;
    uint k_pad0;
    uint k_pad1;
    uint k_pad2;
    uint4 k_optionalParams[4];
};

cbuffer MatV_Params : register(b2)
{
    uint v_dimx;
    uint v_dimy;
    uint v_dimz;
    uint v_offset;
    uint v_uniqueId;
    uint v_pad0;
    uint v_pad1;
    uint v_pad2;
    uint4 v_optionalParams[4];
};

cbuffer MatPacked_Params : register(b3)
{
    uint packed_dimx;
    uint packed_dimy;
    uint packed_dimz;
    uint packed_offset;
    uint packed_uniqueId;
    uint packed_pad0;
    uint packed_pad1;
    uint packed_pad2;
    uint4 packed_optionalParams[4];
};

RWStructuredBuffer<float> MatQ : register(u0);
RWStructuredBuffer<float> MatK : register(u1);
RWStructuredBuffer<float> MatV : register(u2);
StructuredBuffer<float> MatPacked : register(t1);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint count = q_dimx * q_dimy * q_dimz;
    if (idx >= count)
    {
        return;
    }

    uint dC = q_dimx;
    uint row = idx / dC;
    uint col = idx - row * dC;
    uint packedStride = dC * 3;
    uint packedBase = packed_offset + row * packedStride;

    uint qIdx = q_offset + idx;
    uint kIdx = k_offset + idx;
    uint vIdx = v_offset + idx;

    MatQ[qIdx] = MatPacked[packedBase + col];
    MatK[kIdx] = MatPacked[packedBase + col + dC];
    MatV[vIdx] = MatPacked[packedBase + col + 2u * dC];
}
