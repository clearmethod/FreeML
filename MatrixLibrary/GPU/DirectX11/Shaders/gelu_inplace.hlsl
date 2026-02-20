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

RWStructuredBuffer<float> MatOut : register(u0);

float ErfApprox(float x)
{
    float t = 1.0f / (1.0f + 0.5f * abs(x));
    float tau = t * exp(-x * x - 1.26551223f
                        + t * (1.00002368f
                        + t * (0.37409196f
                        + t * (0.09678418f
                        + t * (-0.18628806f
                        + t * (0.27886807f
                        + t * (-1.13520398f
                        + t * (1.48851587f
                        + t * (-0.82215223f
                        + t * 0.17087277f)))))))));
    return (x >= 0.0f) ? (1.0f - tau) : (tau - 1.0f);
}

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint out_plane = out_dimx * out_dimy;
    uint out_count = out_plane * out_dimz;
    if (idx >= out_count)
        return;

    uint out_x = idx % out_dimx;
    uint out_y = (idx / out_dimx) % out_dimy;
    uint out_z = idx / out_plane;

    uint out_base = out_offset + out_z * out_plane;
    uint out_idx = out_base + (out_y * out_dimx) + out_x;

    float x = MatOut[out_idx];
    float erfTerm = ErfApprox(x * 0.70710678118f);
    MatOut[out_idx] = 0.5f * x * (1.0f + erfTerm);
}
