// VAE reparameterization backward pass.
//
// Given dz (upstream gradient), mu, logvar, eps (saved from forward),
// computes per-element gradients for both the mu and logvar heads:
//
//   lv    = clamp(logvar[i], -10, 4)
//   sigma = exp(0.5 * lv)
//   muGrad[i]  = dz[i] - kl_weight * mu[i]
//   lvGrad[i]  = dz[i] * eps[i] * 0.5 * sigma
//                - kl_weight * 0.5 * (sigma^2 - 1)

cbuffer MuGrad_Params : register(b0)
{
    uint muGrad_dimx;
    uint muGrad_dimy;
    uint muGrad_dimz;
    uint muGrad_offset;
    uint muGrad_uniqueId;
    uint muGrad_pad0;
    uint muGrad_pad1;
    uint muGrad_pad2;
    uint kl_weight_bits; // asfloat(kl_weight_bits) = kl_weight
    uint muGrad_opt1;
    uint muGrad_opt2;
    uint muGrad_opt3;
    uint muGrad_opt4;
    uint muGrad_opt5;
    uint muGrad_opt6;
    uint muGrad_opt7;
    uint muGrad_opt8;
    uint muGrad_opt9;
    uint muGrad_opt10;
    uint muGrad_opt11;
    uint muGrad_opt12;
    uint muGrad_opt13;
    uint muGrad_opt14;
    uint muGrad_opt15;
};

cbuffer LvGrad_Params : register(b1)
{
    uint lvGrad_dimx;
    uint lvGrad_dimy;
    uint lvGrad_dimz;
    uint lvGrad_offset;
    uint lvGrad_uniqueId;
    uint lvGrad_pad0;
    uint lvGrad_pad1;
    uint lvGrad_pad2;
};

cbuffer DZ_Params : register(b2)
{
    uint dz_dimx;
    uint dz_dimy;
    uint dz_dimz;
    uint dz_offset;
    uint dz_uniqueId;
    uint dz_pad0;
    uint dz_pad1;
    uint dz_pad2;
};

cbuffer Mu_Params : register(b3)
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

cbuffer LogVar_Params : register(b4)
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

cbuffer Eps_Params : register(b5)
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

RWStructuredBuffer<float> MuGrad  : register(u0);
RWStructuredBuffer<float> LvGrad  : register(u1);
StructuredBuffer<float>   DZ      : register(t1);
StructuredBuffer<float>   Mu      : register(t2);
StructuredBuffer<float>   LogVar  : register(t3);
StructuredBuffer<float>   Eps     : register(t4);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx   = dispatchThreadId.x;
    uint count = muGrad_dimx * muGrad_dimy * muGrad_dimz;
    if (idx >= count)
        return;

    float klw   = asfloat(kl_weight_bits);
    float dz    = DZ    [dz_offset     + idx];
    float mu    = Mu    [mu_offset     + idx];
    float lv    = clamp(LogVar[lv_offset + idx], -10.0f, 4.0f);
    float e     = Eps   [eps_offset    + idx];
    float sigma = exp(0.5f * lv);

    MuGrad[muGrad_offset + idx] = dz - klw * mu;
    LvGrad[lvGrad_offset + idx] = dz * e * 0.5f * sigma
                                  - klw * 0.5f * (sigma * sigma - 1.0f);
}
