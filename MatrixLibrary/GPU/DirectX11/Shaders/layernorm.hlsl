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

cbuffer MatGamma_Params : register(b2)
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

cbuffer MatBeta_Params : register(b3)
{
    uint beta_dimx;
    uint beta_dimy;
    uint beta_dimz;
    uint beta_offset;
    uint beta_uniqueId;
    uint beta_pad0;
    uint beta_pad1;
    uint beta_pad2;
};

cbuffer MatXHat_Params : register(b4)
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

RWStructuredBuffer<float> MatOut  : register(u0);
RWStructuredBuffer<float> MatXHat : register(u1);
StructuredBuffer<float>   MatIn    : register(t1);
StructuredBuffer<float>   MatGamma : register(t2);
StructuredBuffer<float>   MatBeta  : register(t3);

#define GROUP_SIZE 256
groupshared float gs_buf[GROUP_SIZE];

// One group per row. Each thread covers ceil(N / GROUP_SIZE) elements.
// Dispatch: (dimy * dimz) groups, 1 thread group per row.
[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint3 groupId     : SV_GroupID,
            uint3 threadId    : SV_GroupThreadID)
{
    uint tid = threadId.x;
    uint row = groupId.x;

    uint rows = out_dimy * out_dimz;
    if (row >= rows)
        return;

    uint out_z = row / out_dimy;
    uint out_y = row - out_z * out_dimy;
    uint N = in_dimx;

    uint in_plane   = in_dimx   * in_dimy;
    uint out_plane  = out_dimx  * out_dimy;
    uint xhat_plane = xhat_dimx * xhat_dimy;

    uint in_row   = in_offset   + out_z * in_plane   + out_y * in_dimx;
    uint out_row  = out_offset  + out_z * out_plane  + out_y * out_dimx;
    uint xhat_row = xhat_offset + out_z * xhat_plane + out_y * xhat_dimx;

    // Pass 1: mean — thread 0 sums sequentially to match the CPU's left-to-right
    // accumulation order and preserve float parity with the CPU reference.
    if (tid == 0)
    {
        float sum = 0.0f;
        for (uint i = 0; i < N; i++)
            sum += MatIn[in_row + i];
        gs_buf[0] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    float mean = gs_buf[0] / (float)N;
    GroupMemoryBarrierWithGroupSync();  // All threads read gs_buf[0] before thread 0 overwrites it.

    // Pass 2: variance — thread 0 sums sequentially (same ordering rationale).
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
    float invStd = rsqrt(gs_buf[0] / (float)N + 1e-5f);

    // Pass 3: write normalised output and xHat.
    for (uint i = tid; i < N; i += GROUP_SIZE)
    {
        float normalized = (MatIn[in_row + i] - mean) * invStd;
        MatXHat[xhat_row + i] = normalized;
        MatOut[out_row + i]   = normalized * MatGamma[gamma_offset + i]
                              + MatBeta[beta_offset + i];
    }
}
