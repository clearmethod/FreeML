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

RWStructuredBuffer<float> MatDX    : register(u0);
StructuredBuffer<float>   MatDY    : register(t1);
StructuredBuffer<float>   MatIn    : register(t2);
StructuredBuffer<float>   MatGamma : register(t3);

// Shared memory only needs index 0 for each scalar broadcast.
#define GROUP_SIZE 256
groupshared float gs_buf[GROUP_SIZE];
groupshared float gs_buf2[GROUP_SIZE];
groupshared float gs_buf3[GROUP_SIZE];

// Matches the standard LayerNorm backward formula used in the CPU implementation.
// Thread 0 performs each sequential reduction to preserve the same floating-point
// summation order as the CPU reference, ensuring CPU/GPU parity.
//
//   mean   = sum(x_i) / N
//   var    = sum((x_i - mean)^2) / N
//   invStd = 1 / sqrt(var + eps)
//   dXhat_i = dY_i * gamma_i
//   dVar  = sum_j( dXhat_j * (x_j - mean) ) * (-0.5) * invStd^3
//   dMean = sum_j( dXhat_j ) * (-invStd)  +  dVar * (-2/N) * sum_j(x_j - mean)
//   dX_i  = dXhat_i * invStd  +  dVar * 2 * (x_i - mean) / N  +  dMean / N
//
// Dispatch: (dimy * dimz) groups, one group per row.
[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint3 groupId  : SV_GroupID,
            uint3 threadId : SV_GroupThreadID)
{
    uint tid = threadId.x;
    uint row = groupId.x;

    uint rows = dx_dimy * dx_dimz;
    if (row >= rows)
        return;

    uint out_z = row / dx_dimy;
    uint out_y = row - out_z * dx_dimy;
    uint N = dx_dimx;

    uint dx_plane = dx_dimx * dx_dimy;
    uint dy_plane = dy_dimx * dy_dimy;
    uint in_plane = in_dimx * in_dimy;

    uint dx_row = dx_offset + out_z * dx_plane + out_y * dx_dimx;
    uint dy_row = dy_offset + out_z * dy_plane + out_y * dy_dimx;
    uint in_row = in_offset + out_z * in_plane + out_y * in_dimx;

    // Pass 1: mean — thread 0 sums sequentially to match CPU float order.
    if (tid == 0)
    {
        float sum = 0.0f;
        for (uint i = 0; i < N; i++)
            sum += MatIn[in_row + i];
        gs_buf[0] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    float mean = gs_buf[0] / (float)N;
    GroupMemoryBarrierWithGroupSync();

    // Pass 2: variance → invStd — thread 0 sums sequentially.
    if (tid == 0)
    {
        float var = 0.0f;
        for (uint i = 0; i < N; i++)
        {
            float diff = MatIn[in_row + i] - mean;
            var += diff * diff;
        }
        gs_buf[0] = var;
    }
    GroupMemoryBarrierWithGroupSync();
    float invStd  = rsqrt(gs_buf[0] / (float)N + 1e-5f);
    float invStd3 = invStd * invStd * invStd;
    float invN    = 1.0f / (float)N;
    GroupMemoryBarrierWithGroupSync();

    // Pass 3: dVar, dMean, sumDiff — thread 0 accumulates sequentially.
    if (tid == 0)
    {
        float dVar_   = 0.0f;
        float dMean_  = 0.0f;
        float sumDiff = 0.0f;
        for (uint i = 0; i < N; i++)
        {
            float diff  = MatIn[in_row + i] - mean;
            float dXhat = MatDY[dy_row + i] * MatGamma[gamma_offset + i];
            dVar_   += dXhat * diff * (-0.5f) * invStd3;
            dMean_  += dXhat * (-invStd);
            sumDiff += diff;
        }
        gs_buf[0]  = dVar_;
        gs_buf2[0] = dMean_;
        gs_buf3[0] = sumDiff;
    }
    GroupMemoryBarrierWithGroupSync();
    float dVar  = gs_buf[0];
    float dMean = gs_buf2[0] + dVar * (-2.0f) * gs_buf3[0] * invN;

    // Pass 4: write dX — all threads in parallel.
    for (uint i = tid; i < N; i += GROUP_SIZE)
    {
        float diff  = MatIn[in_row + i] - mean;
        float dXhat = MatDY[dy_row + i] * MatGamma[gamma_offset + i];
        MatDX[dx_row + i] = dXhat * invStd
                          + dVar  * 2.0f * diff * invN
                          + dMean * invN;
    }
}
