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

[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint row = dispatchThreadId.x;
    uint rows = out_dimy * out_dimz;
    if (row >= rows)
        return;

    uint out_plane = out_dimx * out_dimy;
    uint out_z = row / out_dimy;
    uint out_y = row - (out_z * out_dimy);

    uint out_base = out_offset + out_z * out_plane + out_y * out_dimx;
    uint in_plane = in_dimx * in_dimy;
    uint in_base = in_offset + out_z * in_plane + out_y * in_dimx;

    float maxVal = MatIn[in_base];
    for (uint x = 1; x < out_dimx; ++x)
    {
        float v = MatIn[in_base + x];
        maxVal = (v > maxVal) ? v : maxVal;
    }

    float sum = 0.0f;
    for (uint x = 0; x < out_dimx; ++x)
    {
        sum += exp(MatIn[in_base + x] - maxVal);
    }

    for (uint x = 0; x < out_dimx; ++x)
    {
        MatOut[out_base + x] = exp(MatIn[in_base + x] - maxVal) / sum;
    }
}
