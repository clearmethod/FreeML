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
    const uint localId = groupThreadId.x;
    const uint zLayer = out_optionalParams[0].x;
    const uint outX = out_optionalParams[0].y;
    const uint outY = out_optionalParams[0].z;
    const uint outZ = out_optionalParams[0].w;

    const uint spatialCount = in_dimx * in_dimy;
    const uint inBase = in_offset + zLayer * spatialCount;

    float localSum = 0.0f;
    for (uint i = localId; i < spatialCount; i += GROUP_SIZE)
    {
        localSum += MatIn[inBase + i];
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
        const uint outPlane = out_dimx * out_dimy;
        const uint outIndex = out_offset + outZ * outPlane + outY * out_dimx + outX;
        MatOut[outIndex] = gs_buf[0];
    }
}
