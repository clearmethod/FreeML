cbuffer MatDX_Params : register(b0)
{
    uint dx_dimx;
    uint dx_dimy;
    uint dx_dimz;
    uint dx_offset;
    uint dx_uniqueId;
    uint dx_pad0;
    uint dx_pad1;
    uint dx_pad2;
};

cbuffer MatDY_Params : register(b1)
{
    uint dy_dimx;
    uint dy_dimy;
    uint dy_dimz;
    uint dy_offset;
    uint dy_uniqueId;
    uint dy_pad0;
    uint dy_pad1;
    uint dy_pad2;
};

cbuffer MatIn_Params : register(b2)
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

cbuffer MatGamma_Params : register(b3)
{
    uint gamma_dimx;
    uint gamma_dimy;
    uint gamma_dimz;
    uint gamma_offset;
    uint gamma_uniqueId;
    uint gamma_pad0;
    uint gamma_pad1;
    uint gamma_pad2;
};

RWStructuredBuffer<float> MatDX : register(u0);
StructuredBuffer<float> MatDY : register(t1);
StructuredBuffer<float> MatIn : register(t2);
StructuredBuffer<float> MatGamma : register(t3);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint dx_plane = dx_dimx * dx_dimy;
    uint dx_count = dx_plane * dx_dimz;
    if (idx >= dx_count)
        return;

    uint out_x = idx % dx_dimx;
    uint out_y = (idx / dx_dimx) % dx_dimy;
    uint out_z = idx / dx_plane;

    uint in_plane = in_dimx * in_dimy;
    uint dy_plane = dy_dimx * dy_dimy;
    uint row_base_in = in_offset + out_z * in_plane + out_y * in_dimx;
    uint row_base_dy = dy_offset + out_z * dy_plane + out_y * dy_dimx;

    float mean = 0.0f;
    for (uint i = 0; i < in_dimx; ++i)
    {
        mean += MatIn[row_base_in + i];
    }
    mean /= (float)in_dimx;

    float var = 0.0f;
    for (uint i = 0; i < in_dimx; ++i)
    {
        float diff = MatIn[row_base_in + i] - mean;
        var += diff * diff;
    }
    var /= (float)in_dimx;

    float invStd = rsqrt(var + 1e-5f);
    float invStd3 = invStd * invStd * invStd;

    float dVar = 0.0f;
    float dMean = 0.0f;
    float sumDiff = 0.0f;
    for (uint i = 0; i < in_dimx; ++i)
    {
        float diff = MatIn[row_base_in + i] - mean;
        float dY = MatDY[row_base_dy + i];
        float dXhat = dY * MatGamma[gamma_offset + i];

        dVar += dXhat * diff * -0.5f * invStd3;
        dMean += dXhat * -invStd;
        sumDiff += diff;
    }
    dMean += dVar * -2.0f * sumDiff / (float)in_dimx;

    float diff = MatIn[row_base_in + out_x] - mean;
    float dXhat = MatDY[row_base_dy + out_x] * MatGamma[gamma_offset + out_x];
    float dX = dXhat * invStd
               + dVar * 2.0f * diff / (float)in_dimx
               + dMean / (float)in_dimx;

    uint dx_base = dx_offset + out_z * dx_plane + out_y * dx_dimx;
    MatDX[dx_base + out_x] = dX;
}
