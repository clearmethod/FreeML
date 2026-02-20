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
};

cbuffer MatMask_Params : register(b1)
{
    uint mask_dimx;
    uint mask_dimy;
    uint mask_dimz;
    uint mask_offset;
    uint mask_uniqueId;
    uint mask_pad0;
    uint mask_pad1;
    uint mask_pad2;
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatMask : register(t1);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint out_plane = out_dimx * out_dimy;
    uint out_count = out_plane * out_dimz;
    if (idx >= out_count)
        return;

    uint out_x = idx % out_dimx;
    uint out_y = (idx / out_dimx) % out_dimy;
    uint out_z = idx / out_plane;

    uint mask_plane = mask_dimx * mask_dimy;

    uint out_base = out_offset + out_z * out_plane;
    uint mask_base = mask_offset + out_z * mask_plane;

    uint out_idx = out_base + (out_y * out_dimx) + out_x;
    uint mask_idx = mask_base + (out_y * mask_dimx) + out_x;

    MatOut[out_idx] = MatOut[out_idx] * MatMask[mask_idx];
}
