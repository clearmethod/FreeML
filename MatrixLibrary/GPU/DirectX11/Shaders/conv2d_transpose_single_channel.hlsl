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

RWStructuredBuffer<float> MatOut : register(u0);
StructuredBuffer<float> MatIn : register(t1);
StructuredBuffer<float> MatKernel : register(t2);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint outCount = out_dimx * out_dimy;
    if (idx >= outCount)
        return;

    const uint outputChannel = out_optionalParams[0].x;
    const int stride = int(out_optionalParams[0].y);
    const int dilation = int(out_optionalParams[0].z);
    const int padding = int(out_optionalParams[0].w);

    const uint x = idx % out_dimx;
    const uint y = idx / out_dimx;
    const uint outPlane = out_dimx * out_dimy;
    const uint inputPlane = in_dimx * in_dimy;
    const uint kernelPlane = kernel_dimx * kernel_dimy;

    float sum = 0.0f;
    for (uint ic = 0u; ic < in_dimz; ++ic)
    {
        for (uint ky = 0u; ky < kernel_dimy; ++ky)
        {
            for (uint kx = 0u; kx < kernel_dimx; ++kx)
            {
                const int numX = int(x) + padding - int(kx) * dilation;
                const int numY = int(y) + padding - int(ky) * dilation;
                if (numX < 0 || numY < 0)
                    continue;
                if ((numX % stride) != 0 || (numY % stride) != 0)
                    continue;

                const uint inX = uint(numX / stride);
                const uint inY = uint(numY / stride);
                if (inX >= in_dimx || inY >= in_dimy)
                    continue;

                const uint inputIdx = in_offset + ic * inputPlane + inY * in_dimx + inX;
                const uint kernelIdx = kernel_offset + ic * kernelPlane + ky * kernel_dimx + kx;
                sum += MatIn[inputIdx] * MatKernel[kernelIdx];
            }
        }
    }

    const uint outIdx = out_offset + outputChannel * outPlane + y * out_dimx + x;
    MatOut[outIdx] = sum;
}
