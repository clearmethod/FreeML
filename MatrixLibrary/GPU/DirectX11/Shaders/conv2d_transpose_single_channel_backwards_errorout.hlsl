cbuffer MatErrorIn_Params : register(b0)
{
    uint error_dimx;
    uint error_dimy;
    uint error_dimz;
    uint error_offset;
    uint error_uniqueId;
    uint error_channels;
    uint error_batch;
    uint error_reserved;
    uint4 error_optionalParams[4];
};

cbuffer MatInput_Params : register(b1)
{
    uint input_dimx;
    uint input_dimy;
    uint input_dimz;
    uint input_offset;
    uint input_uniqueId;
    uint input_channels;
    uint input_batch;
    uint input_reserved;
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

cbuffer MatErrorOut_Params : register(b3)
{
    uint errorout_dimx;
    uint errorout_dimy;
    uint errorout_dimz;
    uint errorout_offset;
    uint errorout_uniqueId;
    uint errorout_channels;
    uint errorout_batch;
    uint errorout_reserved;
};

RWStructuredBuffer<float> MatErrorOut : register(u0);
StructuredBuffer<float> MatErrorIn : register(t1);
StructuredBuffer<float> MatInput : register(t2);
StructuredBuffer<float> MatKernel : register(t3);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    const uint count = errorout_dimx * errorout_dimy * errorout_dimz;
    if (idx >= count)
        return;

    const uint outputChannel = error_optionalParams[0].x;
    const int stride = int(error_optionalParams[0].y);
    const int dilation = int(error_optionalParams[0].z);
    const int padding = int(error_optionalParams[0].w);

    const uint errorOutPlane = errorout_dimx * errorout_dimy;
    const uint plane = idx / errorOutPlane;
    const uint ic = plane % errorout_channels, batch = plane / errorout_channels;
    const uint errorOutPlaneIndex = idx - plane * errorOutPlane;
    const uint inY = errorOutPlaneIndex / errorout_dimx;
    const uint inX = errorOutPlaneIndex - inY * errorout_dimx;

    float sum = 0.0f;
    const uint kernelPlane = kernel_dimx * kernel_dimy;
    const uint errorPlane = error_dimx * error_dimy;
    for (uint ky = 0u; ky < kernel_dimy; ++ky)
    {
        for (uint kx = 0u; kx < kernel_dimx; ++kx)
        {
            const int outX = int(inX) * stride + int(kx) * dilation - padding;
            const int outY = int(inY) * stride + int(ky) * dilation - padding;
            if (outX < 0 || outY < 0 || outX >= int(error_dimx) || outY >= int(error_dimy))
                continue;

            const uint kernelIdx = kernel_offset + ic * kernelPlane + ky * kernel_dimx + kx;
            const uint errorIdx = error_offset + (batch * error_channels + outputChannel) * errorPlane + uint(outY) * error_dimx + uint(outX);
            sum += MatKernel[kernelIdx] * MatErrorIn[errorIdx];
        }
    }

    MatErrorOut[errorout_offset + idx] += sum;
}
