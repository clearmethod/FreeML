cbuffer MatDGamma_Params : register(b0)
{
    uint dgamma_dimx;
    uint dgamma_dimy;
    uint dgamma_dimz;
    uint dgamma_offset;
    uint dgamma_uniqueId;
    uint dgamma_pad0;
    uint dgamma_pad1;
    uint dgamma_pad2;
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

cbuffer MatXHat_Params : register(b2)
{
    uint xhat_dimx;
    uint xhat_dimy;
    uint xhat_dimz;
    uint xhat_offset;
    uint xhat_uniqueId;
    uint xhat_pad0;
    uint xhat_pad1;
    uint xhat_pad2;
};

RWStructuredBuffer<float> MatDGamma : register(u0);
StructuredBuffer<float> MatDY : register(t1);
StructuredBuffer<float> MatXHat : register(t2);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint col = dispatchThreadId.x;
    if (col >= dgamma_dimx)
        return;

    uint rows = dy_dimy * dy_dimz;
    uint dy_plane = dy_dimx * dy_dimy;
    uint xhat_plane = xhat_dimx * xhat_dimy;

    float sumDyXhat = 0.0f;
    for (uint row = 0; row < rows; ++row)
    {
        uint z = row / dy_dimy;
        uint y = row - (z * dy_dimy);
        uint dy_base = dy_offset + z * dy_plane + y * dy_dimx;
        uint xhat_base = xhat_offset + z * xhat_plane + y * xhat_dimx;
        float dy = MatDY[dy_base + col];
        sumDyXhat += dy * MatXHat[xhat_base + col];
    }

    MatDGamma[dgamma_offset + col] = sumDyXhat;
}
