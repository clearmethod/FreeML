#include <metal_stdlib>
using namespace metal;

struct MatrixParams
{
    uint dimx, dimy, dimz, offset;
    uint uniqueId, channels, batch, reserved;
    uint optionalParams[16];
};

// One group computes a 16x16 output tile. Both input tiles are reused by
// all lanes, including for transposed and physically strided head views.
kernel void mul_tiled(device float* out [[buffer(0)]],
                      device const float* left [[buffer(1)]],
                      device const float* right [[buffer(2)]],
                      constant MatrixParams& o [[buffer(3)]],
                      constant MatrixParams& l [[buffer(4)]],
                      constant MatrixParams& r [[buffer(5)]],
                      constant uint& flags [[buffer(6)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint3 lane [[thread_position_in_threadgroup]])
{
    const bool transposeL = (flags & 1u) != 0;
    const bool transposeR = (flags & 2u) != 0;
    const bool strided = (flags & 4u) != 0;
    const uint lr = strided && l.optionalParams[0] ? l.optionalParams[0] : l.dimx;
    const uint lc = strided && l.optionalParams[1] ? l.optionalParams[1] : 1u;
    const uint rr = strided && r.optionalParams[0] ? r.optionalParams[0] : r.dimx;
    const uint rc = strided && r.optionalParams[1] ? r.optionalParams[1] : 1u;
    const uint orow = strided && o.optionalParams[0] ? o.optionalParams[0] : o.dimx;
    const uint ocol = strided && o.optionalParams[1] ? o.optionalParams[1] : 1u;
    const uint lb = l.offset + (l.batch == 1u ? 0u : group.z / o.channels) * (strided ? l.optionalParams[3] : lr * l.dimy * l.channels) + (l.channels == 1u ? 0u : group.z % o.channels) * (strided ? l.optionalParams[2] : lr * l.dimy);
    const uint rb = r.offset + (r.batch == 1u ? 0u : group.z / o.channels) * (strided ? r.optionalParams[3] : rr * r.dimy * r.channels) + (r.channels == 1u ? 0u : group.z % o.channels) * (strided ? r.optionalParams[2] : rr * r.dimy);
    const uint ob = o.offset + (o.batch == 1u ? 0u : group.z / o.channels) * (strided ? o.optionalParams[3] : orow * o.dimy * o.channels) + (o.channels == 1u ? 0u : group.z % o.channels) * (strided ? o.optionalParams[2] : orow * o.dimy);
    const uint row = group.y * 16u + lane.y;
    const uint col = group.x * 16u + lane.x;
    const uint inner = transposeL ? l.dimy : l.dimx;

    threadgroup float a[16][16];
    threadgroup float b[16][16];
    float sum = 0.0f;
    for (uint start = 0; start < inner; start += 16u)
    {
        const uint ak = start + lane.x;
        const uint bk = start + lane.y;
        // Transpose while loading shared tiles so adjacent x lanes still read
        // adjacent physical columns in row-major input buffers.
        if (transposeL)
        {
            const uint loadRow = group.y * 16u + lane.x;
            a[lane.x][lane.y] = loadRow < o.dimy && bk < inner
                ? left[lb + bk * lr + loadRow * lc] : 0.0f;
        }
        else
            a[lane.y][lane.x] = row < o.dimy && ak < inner
                ? left[lb + row * lr + ak * lc] : 0.0f;
        if (transposeR)
        {
            const uint loadCol = group.x * 16u + lane.y;
            b[lane.x][lane.y] = loadCol < o.dimx && ak < inner
                ? right[rb + loadCol * rr + ak * rc] : 0.0f;
        }
        else
            b[lane.y][lane.x] = col < o.dimx && bk < inner
                ? right[rb + bk * rr + col * rc] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16u; ++k)
            sum += a[lane.y][k] * b[k][lane.x];
        // All lanes finish reading before the next tile overwrites shared data.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < o.dimy && col < o.dimx)
        out[ob + row * orow + col * ocol] = sum;
}
