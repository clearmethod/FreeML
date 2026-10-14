#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct MatrixParams
{
    uint dimx, dimy, dimz, offset;
    uint uniqueId, pad0, pad1, pad2;
    uint optionalParams[16];
};

// One group computes a 32x32 output tile. Both input tiles are reused by
// all lanes, including for transposed and physically strided head views.
kernel void mul_simdgroup(device float* out [[buffer(0)]],
                      device const float* left [[buffer(1)]],
                      device const float* right [[buffer(2)]],
                      constant MatrixParams& o [[buffer(3)]],
                      constant MatrixParams& l [[buffer(4)]],
                      constant MatrixParams& r [[buffer(5)]],
                      constant uint& flags [[buffer(6)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]])
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
    const uint lb = l.offset + group.z * (strided ? l.optionalParams[2] : lr * l.dimy);
    const uint rb = r.offset + group.z * (strided ? r.optionalParams[2] : rr * r.dimy);
    const uint ob = o.offset + group.z * (strided ? o.optionalParams[2] : orow * o.dimy);
    // Four 32-lane SIMD groups each accumulate a 16x16 quadrant.
    // Stage transposes/strides and partial edges into dense, zero-padded tiles.
    const uint inner = transposeL ? l.dimy : l.dimx;
    threadgroup float a[32][32];
    threadgroup float b[32][32];
    threadgroup float c[32][32];
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    const uint sy = (sg / 2u) * 16u;
    const uint sx = (sg % 2u) * 16u;
    for (uint start = 0; start < inner; start += 32u)
    {
        for (uint i = tid; i < 1024u; i += 128u)
        {
            const uint y = i / 32u, x = i % 32u;
            if (transposeL)
            {
                const uint row = group.y * 32u + x, k = start + y;
                a[x][y] = row < o.dimy && k < inner
                    ? left[lb + k * lr + row * lc] : 0.0f;
            }
            else
            {
                const uint row = group.y * 32u + y, k = start + x;
                a[y][x] = row < o.dimy && k < inner
                    ? left[lb + row * lr + k * lc] : 0.0f;
            }
            if (transposeR)
            {
                const uint col = group.x * 32u + y, k = start + x;
                b[x][y] = col < o.dimx && k < inner
                    ? right[rb + col * rr + k * rc] : 0.0f;
            }
            else
            {
                const uint col = group.x * 32u + x, k = start + y;
                b[y][x] = col < o.dimx && k < inner
                    ? right[rb + k * rr + col * rc] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32u; k += 8u)
        {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &a[sy][k], 32u);
            simdgroup_load(a1, &a[sy + 8u][k], 32u);
            simdgroup_load(b0, &b[k][sx], 32u);
            simdgroup_load(b1, &b[k][sx + 8u], 32u);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, &c[sy][sx], 32u);
    simdgroup_store(c01, &c[sy][sx + 8u], 32u);
    simdgroup_store(c10, &c[sy + 8u][sx], 32u);
    simdgroup_store(c11, &c[sy + 8u][sx + 8u], 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024u; i += 128u)
    {
        const uint y = i / 32u, x = i % 32u;
        const uint row = group.y * 32u + y, col = group.x * 32u + x;
        if (row < o.dimy && col < o.dimx)
            out[ob + row * orow + col * ocol] = c[y][x];
    }
}
