cbuffer MatDX_Params : register(b0)
{
    uint dx_dimx;
    uint dx_dimy;
    uint dx_dimz;
    uint dx_offset;
    uint dx_uniqueId;
    uint dx_channels;
    uint dx_batch;
    uint dx_reserved;
};

cbuffer MatDY_Params : register(b1)
{
    uint dy_dimx;
    uint dy_dimy;
    uint dy_dimz;
    uint dy_offset;
    uint dy_uniqueId;
    uint dy_channels;
    uint dy_batch;
    uint dy_reserved;
};

cbuffer MatXHat_Params : register(b2)
{
    uint xhat_dimx;
    uint xhat_dimy;
    uint xhat_dimz;
    uint xhat_offset;
    uint xhat_uniqueId;
    uint xhat_channels;
    uint xhat_batch;
    uint xhat_reserved;
};

cbuffer MatGamma_Params : register(b3)
{
    uint gamma_dimx;
    uint gamma_dimy;
    uint gamma_dimz;
    uint gamma_offset;
    uint gamma_uniqueId;
    uint gamma_channels;
    uint gamma_batch;
    uint gamma_reserved;
};

RWStructuredBuffer<float> MatDX    : register(u0);
StructuredBuffer<float>   MatDY    : register(t1);
StructuredBuffer<float>   MatXHat    : register(t2);
StructuredBuffer<float>   MatGamma : register(t3);

cbuffer MatInvStd_Params : register(b4)
{
    uint invstd_dimx, invstd_dimy, invstd_dimz, invstd_offset;
    uint invstd_uniqueId, invstd_channels, invstd_batch, invstd_reserved;
};
StructuredBuffer<float> MatInvStd : register(t4);
#define GROUP_SIZE 256
groupshared float sums[GROUP_SIZE];
groupshared float sumsHat[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID)
{
    uint row = groupId.x, tid = threadId.x;
    if (row >= dx_dimy * dx_dimz) return;
    uint z = row / dx_dimy, y = row % dx_dimy;
    uint dxRow = dx_offset + z * dx_dimx * dx_dimy + y * dx_dimx;
    uint dyRow = dy_offset + z * dy_dimx * dy_dimy + y * dy_dimx;
    uint hatRow = xhat_offset + z * xhat_dimx * xhat_dimy + y * xhat_dimx;
    float sumGrad = 0, sumGradHat = 0;
    for (uint i = tid; i < dx_dimx; i += GROUP_SIZE)
    {
        float grad = MatDY[dyRow + i] * MatGamma[gamma_offset + i];
        sumGrad += grad;
        sumGradHat += grad * MatXHat[hatRow + i];
    }
    sums[tid] = sumGrad; sumsHat[tid] = sumGradHat;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            sums[tid] += sums[tid + stride];
            sumsHat[tid] += sumsHat[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float meanGrad = sums[0] / float(dx_dimx);
    float meanGradHat = sumsHat[0] / float(dx_dimx);
    float invStd = MatInvStd[invstd_offset + row];
    for (uint i = tid; i < dx_dimx; i += GROUP_SIZE)
        MatDX[dxRow + i] = invStd * (MatDY[dyRow + i] * MatGamma[gamma_offset + i]
                                   - meanGrad - MatXHat[hatRow + i] * meanGradHat);
}
