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

cbuffer MatSrc_Params : register(b1)
{
    uint src_dimx;
    uint src_dimy;
    uint src_dimz;
    uint src_offset;
    uint src_uniqueId;
    uint src_pad0;
    uint src_pad1;
    uint src_pad2;
    uint4 src_optionalParams[4];
};

cbuffer MatIndices_Params : register(b2)
{
    uint idx_dimx;
    uint idx_dimy;
    uint idx_dimz;
    uint idx_offset;
    uint idx_uniqueId;
    uint idx_pad0;
    uint idx_pad1;
    uint idx_pad2;
    uint4 idx_optionalParams[4];
};

RWStructuredBuffer<uint> MatOut : register(u0);
StructuredBuffer<float> MatIndices : register(t0);
StructuredBuffer<float> MatSrc : register(t1);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint srcIndexLinear = dispatchThreadId.x;
    uint srcPlane = src_dimx * src_dimy;
    uint srcCount = srcPlane * src_dimz;
    if (srcIndexLinear >= srcCount)
        return;

    uint srcZ = srcIndexLinear / srcPlane;
    uint srcPlaneIndex = srcIndexLinear - srcZ * srcPlane;
    uint srcRow = srcPlaneIndex / src_dimx;
    uint col = srcPlaneIndex - srcRow * src_dimx;

    uint idxIndex = idx_offset + srcZ * idx_dimx * idx_dimy + srcRow * idx_dimx;
    uint dstRow = (uint)MatIndices[idxIndex];
    if (dstRow >= out_dimy)
        return;

    uint dstZ = (out_dimz == 1u) ? 0u : srcZ;
    uint outPlane = out_dimx * out_dimy;
    uint srcIndex = src_offset + srcIndexLinear;
    uint dstIndex = out_offset + dstZ * outPlane + dstRow * out_dimx + col;
    float addValue = MatSrc[srcIndex];

    uint expected = MatOut[dstIndex];
    for (;;)
    {
        const float currentValue = asfloat(expected);
        const uint desired = asuint(currentValue + addValue);
        uint original = 0u;
        InterlockedCompareExchange(MatOut[dstIndex], expected, desired, original);
        if (original == expected)
        {
            break;
        }
        expected = original;
    }
}
