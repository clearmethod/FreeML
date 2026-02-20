cbuffer Param_Params : register(b0)
{
    uint param_dimx;
    uint param_dimy;
    uint param_dimz;
    uint param_offset;
    uint param_uniqueId;
    uint param_pad0;
    uint param_pad1;
    uint param_pad2;
    uint param_opt0;
    uint param_opt1;
    uint param_opt2;
    uint param_opt3;
    uint param_opt4;
    uint param_opt5;
    uint param_opt6;
    uint param_opt7;
    uint param_opt8;
    uint param_opt9;
    uint param_opt10;
    uint param_opt11;
    uint param_opt12;
    uint param_opt13;
    uint param_opt14;
    uint param_opt15;
};

cbuffer Grad_Params : register(b1)
{
    uint grad_dimx;
    uint grad_dimy;
    uint grad_dimz;
    uint grad_offset;
    uint grad_uniqueId;
    uint grad_pad0;
    uint grad_pad1;
    uint grad_pad2;
    uint grad_opt0;
    uint grad_opt1;
    uint grad_opt2;
    uint grad_opt3;
    uint grad_opt4;
    uint grad_opt5;
    uint grad_opt6;
    uint grad_opt7;
    uint grad_opt8;
    uint grad_opt9;
    uint grad_opt10;
    uint grad_opt11;
    uint grad_opt12;
    uint grad_opt13;
    uint grad_opt14;
    uint grad_opt15;
};

cbuffer Mt_Params : register(b2)
{
    uint mt_dimx;
    uint mt_dimy;
    uint mt_dimz;
    uint mt_offset;
    uint mt_uniqueId;
    uint mt_pad0;
    uint mt_pad1;
    uint mt_pad2;
    uint mt_opt0;
    uint mt_opt1;
    uint mt_opt2;
    uint mt_opt3;
    uint mt_opt4;
    uint mt_opt5;
    uint mt_opt6;
    uint mt_opt7;
    uint mt_opt8;
    uint mt_opt9;
    uint mt_opt10;
    uint mt_opt11;
    uint mt_opt12;
    uint mt_opt13;
    uint mt_opt14;
    uint mt_opt15;
};

cbuffer Vt_Params : register(b3)
{
    uint vt_dimx;
    uint vt_dimy;
    uint vt_dimz;
    uint vt_offset;
    uint vt_uniqueId;
    uint vt_pad0;
    uint vt_pad1;
    uint vt_pad2;
    uint vt_opt0;
    uint vt_opt1;
    uint vt_opt2;
    uint vt_opt3;
    uint vt_opt4;
    uint vt_opt5;
    uint vt_opt6;
    uint vt_opt7;
    uint vt_opt8;
    uint vt_opt9;
    uint vt_opt10;
    uint vt_opt11;
    uint vt_opt12;
    uint vt_opt13;
    uint vt_opt14;
    uint vt_opt15;
};

RWStructuredBuffer<float> Param : register(u0);
RWStructuredBuffer<float> Mt : register(u1);
RWStructuredBuffer<float> Vt : register(u2);
StructuredBuffer<float> Grad : register(t1);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint count = param_dimx * param_dimy * param_dimz;
    if (idx >= count)
        return;

    float lr = asfloat(param_opt0);
    float beta1 = asfloat(param_opt1);
    float beta2 = asfloat(param_opt2);
    float beta1_pow_t = asfloat(param_opt3);
    float beta2_pow_t = asfloat(param_opt4);
    float eps = asfloat(param_opt5);
    float weightDecay = asfloat(param_opt6);

    uint pIdx = param_offset + idx;
    uint gIdx = grad_offset + idx;
    uint mIdx = mt_offset + idx;
    uint vIdx = vt_offset + idx;

    float g = Grad[gIdx];
    float mtVal = beta1 * Mt[mIdx] + (1.0f - beta1) * g;
    float vtVal = beta2 * Vt[vIdx] + (1.0f - beta2) * g * g;
    Mt[mIdx] = mtVal;
    Vt[vIdx] = vtVal;

    float mHat = mtVal / (1.0f - beta1_pow_t);
    float vHat = vtVal / (1.0f - beta2_pow_t);
    float p = Param[pIdx];
    float update = lr * mHat / (sqrt(vHat) + eps);
    float decay = lr * weightDecay * p;
    Param[pIdx] = p + update - decay;
}
