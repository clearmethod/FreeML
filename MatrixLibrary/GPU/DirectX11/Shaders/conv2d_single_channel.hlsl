cbuffer MatOut_Params : register(b0)
{
    uint out_dimx;
    uint out_dimy;
    uint out_dimz;
    uint out_offset;
    uint out_uniqueId;
    uint out_channels;
    uint out_batch;
    uint out_reserved;
    uint4 out_optionalParams[4];
};

cbuffer MatIn_Params : register(b1)
{
    uint in_dimx;
    uint in_dimy;
    uint in_dimz;
    uint in_offset;
    uint in_uniqueId;
    uint in_channels;
    uint in_batch;
    uint in_reserved;
};

cbuffer MatKernel_Params : register(b2)
{
    uint kernel_dimx;
    uint kernel_dimy;
    uint kernel_dimz;
    uint kernel_offset;
    uint kernel_uniqueId;
    uint kernel_channels;
    uint kernel_batch;
    uint kernel_reserved;
};

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatIn : register(t1);
StructuredBuffer<float> MatKernel : register(t2);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint outCount = out_dimx * out_dimy * out_batch;
    if (idx >= outCount)
        return;

    const uint outputChannel = out_optionalParams[0].x;
    const uint stride = out_optionalParams[0].y;
    const uint dilation = out_optionalParams[0].z;
    const uint padding = out_optionalParams[0].w;

    uint x = idx % out_dimx;
    uint y = (idx / out_dimx) % out_dimy;
    const uint batch = idx / (out_dimx * out_dimy);

    float sum = 0.0f;
    const uint inputPlane = in_dimx * in_dimy;
    const uint kernelPlane = kernel_dimx * kernel_dimy;
    for (uint kz = 0u; kz < kernel_channels; ++kz)
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
                    const uint inputIdx = in_offset + (batch * in_channels + kz) * inputPlane + uint(inY) * in_dimx + uint(inX);
                    sum += MatKernel[kernelIdx] * MatIn[inputIdx];
                }
            }
        }
    }

    const uint outPlane = out_dimx * out_dimy;
    const uint outIdx = out_offset + (batch * out_channels + outputChannel) * outPlane + y * out_dimx + x;
    MatOut[outIdx] = sum;
}
