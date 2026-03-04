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
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatIn : register(t1);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint outPlane = out_dimx * out_dimy;
    uint outCount = outPlane * out_dimz;
    if (idx >= outCount)
        return;

    bool outIsRightOperand = (out_optionalParams[0].x != 0u);
    uint outIdx = out_offset + idx;
    uint inIdx = in_offset + idx;
    if (outIsRightOperand)
    {
        MatOut[outIdx] = MatIn[inIdx] - MatOut[outIdx];
    }
    else
    {
        MatOut[outIdx] = MatOut[outIdx] - MatIn[inIdx];
    }
}
