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
    uint4 out_optionalParams[4];
};

cbuffer MatTarget_Params : register(b1)
{
    uint target_dimx;
    uint target_dimy;
    uint target_dimz;
    uint target_offset;
    uint target_uniqueId;
    uint target_pad0;
    uint target_pad1;
    uint target_pad2;
    uint4 target_optionalParams[4];
};

cbuffer MatLogits_Params : register(b2)
{
    uint logits_dimx;
    uint logits_dimy;
    uint logits_dimz;
    uint logits_offset;
    uint logits_uniqueId;
    uint logits_pad0;
    uint logits_pad1;
    uint logits_pad2;
    uint4 logits_optionalParams[4];
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatTarget : register(t1);
StructuredBuffer<float> MatLogits : register(t2);

[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint row = dispatchThreadId.x;
    uint rows = logits_dimy * logits_dimz;
    if (row >= rows)
        return;

    uint t = row % logits_dimy;
    uint z = row / logits_dimy;

    uint logits_plane = logits_dimx * logits_dimy;
    uint logits_base = logits_offset + z * logits_plane + t * logits_dimx;

    uint tz = (target_dimz == 1u) ? 0u : z;
    uint target_plane = target_dimx * target_dimy;
    uint target_base = target_offset + tz * target_plane + t * target_dimx;

    float maxLogit = MatLogits[logits_base];
    for (uint x = 1; x < logits_dimx; ++x)
    {
        float v = MatLogits[logits_base + x];
        maxLogit = (v > maxLogit) ? v : maxLogit;
    }

    float sumExp = 0.0f;
    for (uint x = 0; x < logits_dimx; ++x)
    {
        sumExp += exp(MatLogits[logits_base + x] - maxLogit);
    }
    float logSumExp = maxLogit + log(sumExp);

    float loss = 0.0f;
    float valid = 0.0f;
    if (target_dimx == 1u)
    {
        int idx = (int)MatTarget[target_base];
        if (idx >= 0 && (uint)idx < logits_dimx)
        {
            loss = logSumExp - MatLogits[logits_base + (uint)idx];
            valid = 1.0f;
        }
    }
    else
    {
        for (uint x = 0; x < target_dimx; ++x)
        {
            float ti = MatTarget[target_base + x];
            if (ti > 0.5f)
            {
                loss = logSumExp - MatLogits[logits_base + x];
                valid = 1.0f;
                break;
            }
        }
    }

    uint out_base = out_offset + row * out_dimx;
    MatOut[out_base] = loss;
    MatOut[out_base + 1u] = valid;
}
