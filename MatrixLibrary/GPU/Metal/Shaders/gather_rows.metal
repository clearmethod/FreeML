#include <metal_stdlib>
using namespace metal;

struct MatrixParams
{
    uint dimx, dimy, dimz, offset, uniqueId;
    uint channels, batch, reserved;
    uint optionalParams[16];
};

kernel void gather_rows(device float* out [[buffer(0)]],
                        device const float* src [[buffer(1)]],
                        device const float* indices [[buffer(2)]],
                        constant MatrixParams& output [[buffer(3)]],
                        constant MatrixParams& source [[buffer(4)]],
                        constant MatrixParams& index [[buffer(5)]],
                        uint tid [[thread_position_in_grid]])
{
    const uint plane = output.dimx * output.dimy;
    if (tid >= plane * output.dimz) return;
    const uint batch = tid / plane;
    const uint row = (tid / output.dimx) % output.dimy;
    const uint feature = tid % output.dimx;
    const float id = indices[index.offset + batch * index.dimy + row];
    // Valid row IDs follow the CPU gather contract. Guard invalid IDs on GPU
    // to prevent an out-of-bounds access without a CPU index download.
    if (!(id >= 0.0f && id < float(source.dimy)))
    {
        out[output.offset + tid] = 0.0f;
        return;
    }
    const uint sourceBatch = source.dimz == 1u ? 0u : batch;
    out[output.offset + tid] = src[source.offset + sourceBatch * source.dimx * source.dimy
                                   + uint(id) * source.dimx + feature];
}
