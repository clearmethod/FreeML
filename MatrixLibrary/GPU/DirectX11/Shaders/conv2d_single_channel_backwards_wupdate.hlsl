cbuffer MatErrorIn_Params : register(b0)
{
    uint error_dimx;
    uint error_dimy;
    uint error_dimz;
    uint error_offset;
    uint error_uniqueId;
    uint error_pad0;
    uint error_pad1;
    uint error_pad2;
    uint4 error_optionalParams[4];
};

cbuffer MatInput_Params : register(b1)
{
    uint input_dimx;
    uint input_dimy;
    uint input_dimz;
    uint input_offset;
    uint input_uniqueId;
    uint input_pad0;
    uint input_pad1;
    uint input_pad2;
};

cbuffer MatKernel_Params : register(b2)
{
    uint kernel_dimx;
    uint kernel_dimy;
    uint kernel_dimz;
    uint kernel_offset;
    uint kernel_uniqueId;
    uint kernel_pad0;
    uint kernel_pad1;
    uint kernel_pad2;
};

cbuffer MatWUpdate_Params : register(b3)
{
    uint wupdate_dimx;
    uint wupdate_dimy;
    uint wupdate_dimz;
    uint wupdate_offset;
    uint wupdate_uniqueId;
    uint wupdate_pad0;
    uint wupdate_pad1;
    uint wupdate_pad2;
};

RWStructuredBuffer<float> MatWUpdate : register(u0);
StructuredBuffer<float> MatErrorIn : register(t1);
StructuredBuffer<float> MatInput : register(t2);
StructuredBuffer<float> MatKernel : register(t3);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    const uint count = wupdate_dimx * wupdate_dimy * wupdate_dimz;
    if (idx >= count)
        return;

    const uint outputChannel = error_optionalParams[0].x;
    const int stride = int(error_optionalParams[0].y);
    const int dilation = int(error_optionalParams[0].z);
    const int padding = int(error_optionalParams[0].w);

    const uint kernelPlane = wupdate_dimx * wupdate_dimy;
    const uint kz = idx / kernelPlane;
    const uint kernelPlaneIndex = idx - kz * kernelPlane;
    const uint ky = kernelPlaneIndex / wupdate_dimx;
    const uint kx = kernelPlaneIndex - ky * wupdate_dimx;

    float sum = 0.0f;
    const uint inputPlane = input_dimx * input_dimy;
    const uint errorPlane = error_dimx * error_dimy;
    for (uint y = 0u; y < error_dimy; ++y)
    {
        for (uint x = 0u; x < error_dimx; ++x)
        {
            const int inX = int(x) * stride + int(kx) * dilation - padding;
            const int inY = int(y) * stride + int(ky) * dilation - padding;
            if (inX < 0 || inY < 0 || inX >= int(input_dimx) || inY >= int(input_dimy))
                continue;

            const uint inputIdx = input_offset + kz * inputPlane + uint(inY) * input_dimx + uint(inX);
            const uint errorIdx = error_offset + outputChannel * errorPlane + y * error_dimx + x;
            sum += MatInput[inputIdx] * MatErrorIn[errorIdx];
        }
    }

    MatWUpdate[wupdate_offset + idx] += sum;
}
