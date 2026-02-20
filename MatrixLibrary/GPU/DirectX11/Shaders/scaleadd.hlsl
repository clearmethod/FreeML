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

cbuffer MatScale_Params : register(b2)
{
    uint scale_dimx;
    uint scale_dimy;
    uint scale_dimz;
    uint scale_offset;
    uint scale_uniqueId;
    uint scale_pad0;
    uint scale_pad1;
    uint scale_pad2;
};

cbuffer MatAdd_Params : register(b3)
{
    uint add_dimx;
    uint add_dimy;
    uint add_dimz;
    uint add_offset;
    uint add_uniqueId;
    uint add_pad0;
    uint add_pad1;
    uint add_pad2;
};

RWStructuredBuffer<float> MatOut   : register(u0);
StructuredBuffer<float> MatIn   : register(t1);
StructuredBuffer<float> MatScale : register(t2);
StructuredBuffer<float> MatAdd : register(t3);

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

    uint in_plane = in_dimx * in_dimy;
    uint add_plane = add_dimx * add_dimy;

    uint out_base = out_offset + out_z * out_plane;
    uint in_base = in_offset + out_z * in_plane;
    uint add_base = add_offset + out_z * add_plane;

    uint out_idx = out_base + (out_y * out_dimx) + out_x;
    uint in_idx = in_base + (out_y * in_dimx) + out_x;
    uint add_idx = add_base + (out_y * add_dimx) + out_x;

    float scale = MatScale[scale_offset];
    MatOut[out_idx] = MatIn[in_idx] * scale + MatAdd[add_idx];
}
