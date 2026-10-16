cbuffer MatOut_Params : register(b0)
{
    uint out_dimx, out_dimy, out_dimz, out_offset;
    uint out_uniqueId, out_channels, out_batch, out_reserved;
    uint4 out_optionalParams[4]; // [0].x = probability bits, [0].y = seed
};
RWStructuredBuffer<float> MatOut : register(u0);

uint pcg_hash(uint value)
{
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    if (idx >= out_dimx * out_dimy * out_dimz) return;
    float probability = asfloat(out_optionalParams[0].x);
    uint stream = pcg_hash(out_optionalParams[0].y);
    float uniform = float(pcg_hash(idx ^ stream) >> 8u) * (1.0f / 16777216.0f);
    MatOut[out_offset + idx] = uniform < probability ? 0.0f : 1.0f;
}
