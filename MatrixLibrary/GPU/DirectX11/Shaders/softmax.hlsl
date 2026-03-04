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
StructuredBuffer<float>   MatIn  : register(t1);

#define GROUP_SIZE 256
groupshared float gs_buf[GROUP_SIZE];

// One group per row. Dispatch: (dimy * dimz) groups.
// Pass 1 — parallel max reduction (numerical stability).
// Pass 2 — parallel sum(exp(x - max)) reduction.
// Pass 3 — write exp(x - max) / sum.
// exp is recomputed in pass 3 to avoid groupshared storage proportional to N.
[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint3 groupId  : SV_GroupID,
            uint3 threadId : SV_GroupThreadID)
{
    uint tid = threadId.x;
    uint row = groupId.x;

    uint rows = out_dimy * out_dimz;
    if (row >= rows)
        return;

    uint out_z = row / out_dimy;
    uint out_y = row - out_z * out_dimy;
    uint N = out_dimx;

    uint out_plane = out_dimx * out_dimy;
    uint in_plane  = in_dimx  * in_dimy;

    uint out_row = out_offset + out_z * out_plane + out_y * out_dimx;
    uint in_row  = in_offset  + out_z * in_plane  + out_y * in_dimx;

    // Pass 1: find row max.
    float localMax = -1e38f;
    for (uint i = tid; i < N; i += GROUP_SIZE)
    {
        float v = MatIn[in_row + i];
        localMax = (v > localMax) ? v : localMax;
    }
    gs_buf[tid] = localMax;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = GROUP_SIZE / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            gs_buf[tid] = (gs_buf[tid] > gs_buf[tid + s]) ? gs_buf[tid] : gs_buf[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float maxVal = gs_buf[0];
    GroupMemoryBarrierWithGroupSync();

    // Pass 2: sum(exp(x - max)) — thread 0 sums sequentially to match the CPU's
    // left-to-right accumulation order and preserve float parity.
    if (tid == 0)
    {
        float sum = 0.0f;
        for (uint i = 0; i < N; i++)
            sum += exp(MatIn[in_row + i] - maxVal);
        gs_buf[0] = sum;
    }
    GroupMemoryBarrierWithGroupSync();
    float invSum = 1.0f / gs_buf[0];

    // Pass 3: write output — all threads in parallel.
    for (uint i = tid; i < N; i += GROUP_SIZE)
        MatOut[out_row + i] = exp(MatIn[in_row + i] - maxVal) * invSum;
}
