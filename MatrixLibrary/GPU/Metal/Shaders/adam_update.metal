#include <metal_stdlib>
using namespace metal;

struct MatrixParams
{
    uint dimx;
    uint dimy;
    uint dimz;
    uint offset;
    uint uniqueId;
    uint channels;
    uint batch;
    uint reserved;
    uint optionalParams[16];
};

kernel void adam_update(device float* param [[buffer(0)]],
                        device float* mt [[buffer(1)]],
                        device float* vt [[buffer(2)]],
                        device const float* grad [[buffer(3)]],
                        constant MatrixParams& paramParams [[buffer(4)]],
                        constant MatrixParams& gradParams [[buffer(5)]],
                        constant MatrixParams& mtParams [[buffer(6)]],
                        constant MatrixParams& vtParams [[buffer(7)]],
                        uint idx [[thread_position_in_grid]])
{
    const uint count = paramParams.dimx * paramParams.dimy * paramParams.dimz;
    if (idx >= count)
    {
        return;
    }

    const float lr = as_type<float>(paramParams.optionalParams[0]);
    const float beta1 = as_type<float>(paramParams.optionalParams[1]);
    const float beta2 = as_type<float>(paramParams.optionalParams[2]);
    const float beta1PowT = as_type<float>(paramParams.optionalParams[3]);
    const float beta2PowT = as_type<float>(paramParams.optionalParams[4]);
    const float eps = as_type<float>(paramParams.optionalParams[5]);
    const float weightDecay = as_type<float>(paramParams.optionalParams[6]);

    const uint pIdx = paramParams.offset + idx;
    const uint gIdx = gradParams.offset + idx;
    const uint mIdx = mtParams.offset + idx;
    const uint vIdx = vtParams.offset + idx;

    const float g = grad[gIdx];
    const float mtVal = beta1 * mt[mIdx] + (1.0f - beta1) * g;
    const float vtVal = beta2 * vt[vIdx] + (1.0f - beta2) * g * g;
    mt[mIdx] = mtVal;
    vt[vIdx] = vtVal;

    const float mHat = mtVal / (1.0f - beta1PowT);
    const float vHat = vtVal / (1.0f - beta2PowT);
    const float p = param[pIdx];
    const float update = lr * mHat / (sqrt(vHat) + eps);
    const float decay = lr * weightDecay * p;
    param[pIdx] = p + update - decay;
}
