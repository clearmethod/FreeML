#include <metal_stdlib>
using namespace metal;

struct MatrixParams
{
    uint dimx, dimy, dimz, offset, uniqueId;
    uint pad0, pad1, pad2;
    uint optionalParams[16];
};

kernel void scatter_add_rows(device float* out [[buffer(0)]],
                             device const float* src [[buffer(1)]],
                             device const float* indices [[buffer(2)]],
                             constant MatrixParams& output [[buffer(3)]],
                             constant MatrixParams& source [[buffer(4)]],
                             constant MatrixParams& index [[buffer(5)]],
                             uint tid [[thread_position_in_grid]])
{
    if (tid >= output.dimx * output.dimz) return;
    const uint feature = tid % output.dimx;
    const uint destinationBatch = tid / output.dimx;
    const uint firstBatch = output.dimz == 1u ? 0u : destinationBatch;
    const uint endBatch = output.dimz == 1u ? source.dimz : destinationBatch + 1u;
    const uint outputBase = output.offset + destinationBatch * output.dimx * output.dimy;
    for (uint batch = firstBatch; batch < endBatch; ++batch)
    {
        const uint sourceBase = source.offset + batch * source.dimx * source.dimy;
        const uint indexBase = index.offset + batch * index.dimy;
        for (uint row = 0; row < source.dimy; ++row)
        {
            const uint destinationRow = uint(indices[indexBase + row]);
            if (destinationRow < output.dimy)
                out[outputBase + destinationRow * output.dimx + feature] +=
                    src[sourceBase + row * source.dimx + feature];
        }
    }
}
