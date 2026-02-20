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

cbuffer MatLeft_Params : register(b1)
{
    uint left_dimx;
    uint left_dimy;
    uint left_dimz;
    uint left_offset;
    uint left_uniqueId;
    uint left_pad0;
    uint left_pad1;
    uint left_pad2;
};

cbuffer MatRight_Params : register(b2)
{
    uint right_dimx;
    uint right_dimy;
    uint right_dimz;
    uint right_offset;
    uint right_uniqueId;
    uint right_pad0;
    uint right_pad1;
    uint right_pad2;
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatLeft : register(t1);
StructuredBuffer<float> MatRight : register(t2);

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

    uint left_plane = left_dimx * left_dimy;
    uint right_plane = right_dimx * right_dimy;

    uint out_base = out_offset + out_z * out_plane;
    uint left_base = left_offset + out_z * left_plane;
    uint right_base = right_offset + out_z * right_plane;

    uint out_idx = out_base + (out_y * out_dimx) + out_x;
    uint left_idx = left_base + (out_y * left_dimx) + out_x;
    uint right_idx = right_base + (out_y * right_dimx) + out_x;

    MatOut[out_idx] = MatLeft[left_idx] + MatRight[right_idx];
}
