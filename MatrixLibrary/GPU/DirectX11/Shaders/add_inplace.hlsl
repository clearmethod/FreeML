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

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatIn : register(t1);

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
    uint out_base = out_offset + out_z * out_plane;
    uint in_base = in_offset + out_z * in_plane;

    uint out_idx = out_base + (out_y * out_dimx) + out_x;
    uint in_idx = in_base + (out_y * in_dimx) + out_x;

    MatOut[out_idx] = MatOut[out_idx] + MatIn[in_idx];
}
