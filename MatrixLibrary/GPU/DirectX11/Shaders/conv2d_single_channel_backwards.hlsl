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

cbuffer MatErrorOut_Params : register(b4)
{
    uint errorout_dimx;
    uint errorout_dimy;
    uint errorout_dimz;
    uint errorout_offset;
    uint errorout_uniqueId;
    uint errorout_pad0;
    uint errorout_pad1;
    uint errorout_pad2;
};

StructuredBuffer<float> MatErrorIn : register(t1);
StructuredBuffer<float> MatInput : register(t2);
StructuredBuffer<float> MatKernel : register(t3);
RWStructuredBuffer<uint> MatWUpdate : register(u0);
RWStructuredBuffer<uint> MatErrorOut : register(u1);

void AtomicAddFloat(RWStructuredBuffer<uint> buffer, uint index, float value)
{
    uint expected = buffer[index];
    for (;;)
    {
        const float currentValue = asfloat(expected);
        const uint desired = asuint(currentValue + value);
        uint original = 0u;
        InterlockedCompareExchange(buffer[index], expected, desired, original);
        if (original == expected)
        {
            break;
        }
        expected = original;
    }
}

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint errorCount = error_dimx * error_dimy;
    if (idx >= errorCount)
        return;

    const uint outputChannel = error_optionalParams[0].x;
    const uint stride = error_optionalParams[0].y;
    const uint dilation = error_optionalParams[0].z;
    const uint padding = error_optionalParams[0].w;

    uint x = idx % error_dimx;
    uint y = idx / error_dimx;

    const uint errorPlane = error_dimx * error_dimy;
    const uint errorIdx = error_offset + outputChannel * errorPlane + y * error_dimx + x;
    const float grad = MatErrorIn[errorIdx];

    const uint inputPlane = input_dimx * input_dimy;
    const uint kernelPlane = kernel_dimx * kernel_dimy;
    const uint errorOutPlane = errorout_dimx * errorout_dimy;

    for (uint kz = 0u; kz < kernel_dimz; ++kz)
    {
        for (uint ky = 0u; ky < kernel_dimy; ++ky)
        {
            for (uint kx = 0u; kx < kernel_dimx; ++kx)
            {
                const int inX = int(x * stride + kx * dilation) - int(padding);
                const int inY = int(y * stride + ky * dilation) - int(padding);
                if (inX >= 0 && inY >= 0 && inX < int(input_dimx) && inY < int(input_dimy))
                {
                    const uint inputIdx = input_offset + kz * inputPlane + uint(inY) * input_dimx + uint(inX);
                    const uint kernelIdx = kernel_offset + kz * kernelPlane + ky * kernel_dimx + kx;
                    const uint wUpdateIdx = wupdate_offset + kz * kernelPlane + ky * wupdate_dimx + kx;
                    const uint errorOutIdx = errorout_offset + kz * errorOutPlane + uint(inY) * errorout_dimx + uint(inX);

                    AtomicAddFloat(MatWUpdate, wUpdateIdx, MatInput[inputIdx] * grad);
                    AtomicAddFloat(MatErrorOut, errorOutIdx, MatKernel[kernelIdx] * grad);
                }
            }
        }
    }
}
