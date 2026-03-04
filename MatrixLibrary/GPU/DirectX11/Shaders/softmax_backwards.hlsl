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
StructuredBuffer<float>   MatSoft : register(t1);
StructuredBuffer<float>   MatGrad : register(t2);

#define GROUP_SIZE 256
groupshared float gs_buf[GROUP_SIZE];

// One group per row. Dispatch: (dimy * dimz) groups.
// Pass 1 — parallel reduction for dot(grad, softmax).
// Pass 2 — write s * (g - dot).
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

    uint out_plane  = out_dimx  * out_dimy;
    uint soft_plane = soft_dimx * soft_dimy;
    uint grad_plane = grad_dimx * grad_dimy;

    uint out_row  = out_offset  + out_z * out_plane  + out_y * out_dimx;
    uint soft_row = soft_offset + out_z * soft_plane + out_y * soft_dimx;
    uint grad_row = grad_offset + out_z * grad_plane + out_y * grad_dimx;

    // Pass 1: dot(grad, softmax) — thread 0 sums sequentially to match the CPU's
    // left-to-right accumulation order and preserve float parity.
    if (tid == 0)
    {
        float dot = 0.0f;
        for (uint i = 0; i < N; i++)
            dot += MatGrad[grad_row + i] * MatSoft[soft_row + i];
        gs_buf[0] = dot;
    }
    GroupMemoryBarrierWithGroupSync();
    float dot = gs_buf[0];

    // Pass 2: write s * (g - dot).
    for (uint i = tid; i < N; i += GROUP_SIZE)
    {
        float s = MatSoft[soft_row + i];
        float g = MatGrad[grad_row + i];
        MatOut[out_row + i] = s * (g - dot);
    }
}
