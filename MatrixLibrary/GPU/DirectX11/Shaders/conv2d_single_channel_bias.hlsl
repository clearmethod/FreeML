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

cbuffer MatIn_Params : register(b1)
{
    uint in_dimx;
    uint in_dimy;
    uint in_dimz;
    uint in_offset;
    uint in_uniqueId;
    uint in_pad0;
    uint in_pad1;
    uint in_pad2;
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

cbuffer MatBias_Params : register(b3)
{
    uint bias_dimx;
    uint bias_dimy;
    uint bias_dimz;
    uint bias_offset;
    uint bias_uniqueId;
    uint bias_pad0;
    uint bias_pad1;
    uint bias_pad2;
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatIn : register(t1);
StructuredBuffer<float> MatKernel : register(t2);
StructuredBuffer<float> MatBias : register(t3);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint outCount = out_dimx * out_dimy;
    if (idx >= outCount)
        return;

    const uint outputChannel = out_optionalParams[0].x;
    const uint stride = out_optionalParams[0].y;
    const uint dilation = out_optionalParams[0].z;
    const uint padding = out_optionalParams[0].w;

    uint x = idx % out_dimx;
    uint y = idx / out_dimx;

    float sum = 0.0f;
    const uint inputPlane = in_dimx * in_dimy;
    const uint kernelPlane = kernel_dimx * kernel_dimy;
    for (uint kz = 0u; kz < kernel_dimz; ++kz)
    {
        for (uint ky = 0u; ky < kernel_dimy; ++ky)
        {
            for (uint kx = 0u; kx < kernel_dimx; ++kx)
            {
                const int inX = int(x * stride + kx * dilation) - int(padding);
                const int inY = int(y * stride + ky * dilation) - int(padding);
                if (inX >= 0 && inY >= 0 && inX < int(in_dimx) && inY < int(in_dimy))
                {
                    const uint kernelIdx = kernel_offset + kz * kernelPlane + ky * kernel_dimx + kx;
                    const uint inputIdx = in_offset + kz * inputPlane + uint(inY) * in_dimx + uint(inX);
                    sum += MatKernel[kernelIdx] * MatIn[inputIdx];
                }
            }
        }
    }

    sum += MatBias[bias_offset + outputChannel];

    const uint outPlane = out_dimx * out_dimy;
    const uint outIdx = out_offset + outputChannel * outPlane + y * out_dimx + x;
    MatOut[outIdx] = sum;
}
