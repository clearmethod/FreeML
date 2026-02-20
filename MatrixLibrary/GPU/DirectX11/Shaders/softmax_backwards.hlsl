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

cbuffer MatSoft_Params : register(b1)
{
    uint soft_dimx;
    uint soft_dimy;
    uint soft_dimz;
    uint soft_offset;
    uint soft_uniqueId;
    uint soft_pad0;
    uint soft_pad1;
    uint soft_pad2;
};

cbuffer MatGrad_Params : register(b2)
{
    uint grad_dimx;
    uint grad_dimy;
    uint grad_dimz;
    uint grad_offset;
    uint grad_uniqueId;
    uint grad_pad0;
    uint grad_pad1;
    uint grad_pad2;
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatSoft : register(t1);
StructuredBuffer<float> MatGrad : register(t2);

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
    uint soft_plane = soft_dimx * soft_dimy;
    uint grad_plane = grad_dimx * grad_dimy;
    uint soft_base = soft_offset + out_z * soft_plane + out_y * soft_dimx;
    uint grad_base = grad_offset + out_z * grad_plane + out_y * grad_dimx;

    float dot = 0.0f;
    for (uint x = 0; x < out_dimx; ++x)
    {
        dot += MatGrad[grad_base + x] * MatSoft[soft_base + x];
    }

    for (uint x = 0; x < out_dimx; ++x)
    {
        float s = MatSoft[soft_base + x];
        float g = MatGrad[grad_base + x];
        MatOut[out_base + x] = s * (g - dot);
    }
}
