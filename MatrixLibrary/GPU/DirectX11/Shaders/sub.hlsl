cbuffer MatOut_Params : register(b0)
{
    uint out_dimx;
    uint out_dimy;
    uint out_dimz;
    uint out_offset;
    uint out_uniqueId;
    uint out_channels;
    uint out_batch;
    uint out_reserved;
};

cbuffer MatLeft_Params : register(b1)
{
    uint left_dimx;
    uint left_dimy;
    uint left_dimz;
    uint left_offset;
    uint left_uniqueId;
    uint left_channels;
    uint left_batch;
    uint left_reserved;
};

cbuffer MatRight_Params : register(b2)
{
    uint right_dimx;
    uint right_dimy;
    uint right_dimz;
    uint right_offset;
    uint right_uniqueId;
    uint right_channels;
    uint right_batch;
    uint right_reserved;
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatLeft : register(t1);
StructuredBuffer<float> MatRight : register(t2);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint outPlane = out_dimx * out_dimy;
    uint outCount = outPlane * out_dimz;
    if (idx >= outCount)
        return;

    MatOut[out_offset + idx] = MatLeft[left_offset + idx] - MatRight[right_offset + idx];
}
