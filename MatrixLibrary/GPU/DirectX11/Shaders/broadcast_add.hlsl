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

cbuffer MatBias_Params : register(b1)
{
    uint bias_dimx;
    uint bias_dimy;
    uint bias_dimz;
    uint bias_offset;
    uint bias_uniqueId;
    uint bias_pad0;
    uint bias_pad1;
    uint bias_pad2;
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatBias : register(t1);

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

    uint out_base = out_offset + out_z * out_plane;
    uint out_idx = out_base + (out_y * out_dimx) + out_x;
    uint bias_idx = bias_offset + out_x;

    MatOut[out_idx] += MatBias[bias_idx];
}
