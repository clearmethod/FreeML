// Reparameterization trick: z = mu + exp(0.5 * clamp(logvar, -10, 4)) * eps
// eps ~ N(0,I) generated per-element via PCG hash + Box-Muller.
// Writes both latent (z) and eps so the backward pass can reuse eps.

cbuffer MatLatent_Params : register(b0)
{
    uint latent_dimx;
    uint latent_dimy;
    uint latent_dimz;
    uint latent_offset;
    uint latent_uniqueId;
    uint latent_pad0;
    uint latent_pad1;
    uint latent_pad2;
    uint4 latent_optionalParams[4]; // [0].x = seed
};

cbuffer MatMu_Params : register(b1)
{
    uint mu_dimx;
    uint mu_dimy;
    uint mu_dimz;
    uint mu_offset;
    uint mu_uniqueId;
    uint mu_pad0;
    uint mu_pad1;
    uint mu_pad2;
};

cbuffer MatLogVar_Params : register(b2)
{
    uint lv_dimx;
    uint lv_dimy;
    uint lv_dimz;
    uint lv_offset;
    uint lv_uniqueId;
    uint lv_pad0;
    uint lv_pad1;
    uint lv_pad2;
};

cbuffer MatEps_Params : register(b3)
{
    uint eps_dimx;
    uint eps_dimy;
    uint eps_dimz;
    uint eps_offset;
    uint eps_uniqueId;
    uint eps_pad0;
    uint eps_pad1;
    uint eps_pad2;
};

RWStructuredBuffer<float> MatLatent : register(u0);
RWStructuredBuffer<float> MatEps    : register(u1);
StructuredBuffer<float>   MatMu     : register(t1);
StructuredBuffer<float>   MatLogVar : register(t2);

// PCG hash: fast, high-quality per-element randomness.
uint pcg_hash(uint s)
{
    uint state = s * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Map a hash to float [0, 1).
float uint_to_uniform(uint h)
{
    return (h >> 8u) * (1.0f / 16777216.0f);
}

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx   = dispatchThreadId.x;
    uint count = latent_dimx * latent_dimy * latent_dimz;
    if (idx >= count)
        return;

    uint seed = latent_optionalParams[0].x;

    // Box-Muller: two independent uniform samples per element.
    uint h0 = pcg_hash(seed ^ (idx * 2u + 1u));
    uint h1 = pcg_hash(seed ^ (idx * 2u + 2u));
    float u0 = max(uint_to_uniform(h0), 1e-7f); // avoid log(0)
    float u1 = uint_to_uniform(h1);
    float e  = sqrt(-2.0f * log(u0)) * cos(6.28318530718f * u1);

    float lv    = clamp(MatLogVar[lv_offset + idx], -10.0f, 4.0f);
    float sigma = exp(0.5f * lv);

    MatEps[eps_offset + idx]       = e;
    MatLatent[latent_offset + idx] = MatMu[mu_offset + idx] + sigma * e;
}
