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
    uint rows = out_dimy * out_dimz;
    if (row >= rows)
        return;

    uint t = row % out_dimy;
    uint z = row / out_dimy;

    uint out_plane = out_dimx * out_dimy;
    uint out_base = out_offset + z * out_plane + t * out_dimx;

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

    if (target_dimx == 1u)
    {
        int idx = (int)MatTarget[target_base];
        if (idx < 0 || (uint)idx >= logits_dimx)
        {
            for (uint x = 0; x < out_dimx; ++x)
            {
                MatOut[out_base + x] = 0.0f;
            }
            return;
        }

        float invValid = asfloat(out_optionalParams[0].x);
        for (uint x = 0; x < out_dimx; ++x)
        {
            float p = exp(MatLogits[logits_base + x] - maxLogit) / sumExp;
            float ti = (x == (uint)idx) ? 1.0f : 0.0f;
            MatOut[out_base + x] = (ti - p) * invValid;
        }
        return;
    }

    for (uint x = 0; x < out_dimx; ++x)
    {
        float p = exp(MatLogits[logits_base + x] - maxLogit) / sumExp;
        float ti = MatTarget[target_base + x];
        MatOut[out_base + x] = ti - p;
    }
}
