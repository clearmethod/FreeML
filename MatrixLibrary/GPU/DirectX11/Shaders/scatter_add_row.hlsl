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

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatIndices : register(t0);
StructuredBuffer<float> MatSrc : register(t1);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint outIndexLinear = dispatchThreadId.x;
    uint outPlane = out_dimx * out_dimy;
    uint outCount = outPlane * out_dimz;
    if (outIndexLinear >= outCount)
        return;

    uint outZ = outIndexLinear / outPlane;
    uint outPlaneIndex = outIndexLinear - outZ * outPlane;
    uint outRow = outPlaneIndex / out_dimx;
    uint col = outPlaneIndex - outRow * out_dimx;

    uint srcZ = (src_dimz == 1u) ? 0u : outZ;
    uint srcPlane = src_dimx * src_dimy;
    float sum = MatOut[out_offset + outIndexLinear];

    for (uint srcRow = 0u; srcRow < src_dimy; ++srcRow)
    {
        uint idxIndex = idx_offset + srcZ * idx_dimx * idx_dimy + srcRow * idx_dimx;
        uint dstRow = (uint)MatIndices[idxIndex];
        if (dstRow == outRow)
        {
            uint srcIndex = src_offset + srcZ * srcPlane + srcRow * src_dimx + col;
            sum += MatSrc[srcIndex];
        }
    }

    MatOut[out_offset + outIndexLinear] = sum;
}
