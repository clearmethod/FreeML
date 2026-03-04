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

cbuffer MatTarget_Params : register(b1)
{
    uint target_dimx;
    uint target_dimy;
    uint target_dimz;
    uint target_offset;
    uint target_uniqueId;
    uint target_pad0;
    uint target_pad1;
    uint target_pad2;
    uint4 target_optionalParams[4];
};

cbuffer MatPrediction_Params : register(b2)
{
    uint prediction_dimx;
    uint prediction_dimy;
    uint prediction_dimz;
    uint prediction_offset;
    uint prediction_uniqueId;
    uint prediction_pad0;
    uint prediction_pad1;
    uint prediction_pad2;
    uint4 prediction_optionalParams[4];
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatTarget : register(t1);
StructuredBuffer<float> MatPrediction : register(t2);

groupshared float gs_buf[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    uint row = groupId.x;
    uint rows = prediction_dimy * prediction_dimz;
    if (row >= rows)
        return;

    uint localId = groupThreadId.x;
    uint y = row % prediction_dimy;
    uint z = row / prediction_dimy;
    uint tz = (target_dimz == 1u) ? 0u : z;

    uint predictionPlane = prediction_dimx * prediction_dimy;
    uint targetPlane = target_dimx * target_dimy;
    uint predictionBase = prediction_offset + z * predictionPlane + y * prediction_dimx;
    uint targetBase = target_offset + tz * targetPlane + y * target_dimx;

    float localSum = 0.0f;
    for (uint x = localId; x < prediction_dimx; x += GROUP_SIZE)
    {
        float d = MatTarget[targetBase + x] - MatPrediction[predictionBase + x];
        localSum += d * d;
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
        MatOut[out_offset + row] = gs_buf[0];
    }
}
