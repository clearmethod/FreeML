// Fill every element of MatOut with a scalar value.
// The value is passed as a float bit-pattern in out_opt0 (asfloat(out_opt0)).

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
    uint out_opt0; // asfloat(out_opt0) = fill value
    uint out_opt1;
    uint out_opt2;
    uint out_opt3;
    uint out_opt4;
    uint out_opt5;
    uint out_opt6;
    uint out_opt7;
    uint out_opt8;
    uint out_opt9;
    uint out_opt10;
    uint out_opt11;
    uint out_opt12;
    uint out_opt13;
    uint out_opt14;
    uint out_opt15;
};

RWStructuredBuffer<float> MatOut : register(u0);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx   = dispatchThreadId.x;
    uint count = out_dimx * out_dimy * out_dimz;
    if (idx >= count)
        return;

    MatOut[out_offset + idx] = asfloat(out_opt0);
}
