#define GROUP_SIZE 256

cbuffer MatOut_Params : register(b0)
{
    uint out_dimx;
    uint out_dimy;
    uint out_dimz;
    uint out_offset;
    uint out_uniqueId;
    uint out_pad0;
    uint out_pad1;
    uint out_pad2;
    uint4 out_optionalParams[4];
};

cbuffer MatIn_Params : register(b1)
{
    uint in_dimx;
    uint in_dimy;
    uint in_dimz;
    uint in_offset;
    uint in_uniqueId;
    uint in_pad0;
    uint in_pad1;
    uint in_pad2;
    uint4 in_optionalParams[4];
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatIn : register(t1);

groupshared float gs_buf[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint3 groupThreadId : SV_GroupThreadID)
{
    uint localId = groupThreadId.x;
    uint count = in_dimx * in_dimy * in_dimz;

    float localSum = 0.0f;
    for (uint i = localId; i < count; i += GROUP_SIZE)
    {
        localSum += MatIn[in_offset + i];
    }

    gs_buf[localId] = localSum;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = GROUP_SIZE / 2u; stride > 0u; stride >>= 1u)
    {
        if (localId < stride)
        {
            gs_buf[localId] += gs_buf[localId + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localId == 0u)
    {
        MatOut[out_offset] = gs_buf[0];
    }
}
