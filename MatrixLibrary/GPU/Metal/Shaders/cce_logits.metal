#include <metal_stdlib>
using namespace metal;

struct MatrixParams
{
    uint dimx, dimy, dimz, offset;
    uint uniqueId, pad0, pad1, pad2;
};
constant uint groupSize = 256;

inline uint rowBase(constant MatrixParams& p, uint row, uint steps)
{
    const uint z = p.dimz == 1u ? 0u : row / steps;
    return p.offset + (z * p.dimy + row % steps) * p.dimx;
}

inline float2 softmaxStats(device const float* logits, uint base, uint width,
                           uint lane, threadgroup float* shared)
{
    float maximum = -INFINITY;
    for (uint x = lane; x < width; x += groupSize)
        maximum = max(maximum, logits[base + x]);
    shared[lane] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = groupSize / 2; stride; stride /= 2)
    {
        if (lane < stride) shared[lane] = max(shared[lane], shared[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    maximum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint x = lane; x < width; x += groupSize)
        sum += exp(logits[base + x] - maximum);
    shared[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = groupSize / 2; stride; stride /= 2)
    {
        if (lane < stride) shared[lane] += shared[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return float2(maximum, shared[0]);
}

// Count broadcast targets once per logits batch, matching the CPU normalization.
kernel void cce_logits_count(device float* out [[buffer(0)]],
                              device const float* target [[buffer(1)]],
                              constant MatrixParams& o [[buffer(4)]],
                              constant MatrixParams& t [[buffer(5)]],
                              constant MatrixParams& l [[buffer(6)]],
                              uint lane [[thread_position_in_threadgroup]])
{
    threadgroup uint counts[256];
    uint valid = 0;
    for (uint row = lane; row < l.dimy * l.dimz; row += groupSize)
    {
        const int index = int(target[rowBase(t, row, l.dimy)]);
        valid += index >= 0 && uint(index) < l.dimx;
    }
    counts[lane] = valid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = groupSize / 2; stride; stride /= 2)
    {
        if (lane < stride) counts[lane] += counts[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) out[o.offset] = counts[0] ? 1.0f / float(counts[0]) : 0.0f;
}

kernel void cce_logits_grad(device float* out [[buffer(0)]],
                             device const float* target [[buffer(1)]],
                             device const float* logits [[buffer(2)]],
                             device const float* inverseCount [[buffer(3)]],
                             constant MatrixParams& o [[buffer(4)]],
                             constant MatrixParams& t [[buffer(5)]],
                             constant MatrixParams& l [[buffer(6)]],
                             constant MatrixParams& n [[buffer(7)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint lane [[thread_position_in_threadgroup]])
{
    const uint row = group;
    const uint targetBase = rowBase(t, row, l.dimy);
    const uint outputBase = o.offset + row * o.dimx;
    const bool indexed = t.dimx == 1u;
    const int index = indexed ? int(target[targetBase]) : 0;
    if (indexed && (index < 0 || uint(index) >= l.dimx))
    {
        for (uint x = lane; x < l.dimx; x += groupSize) out[outputBase + x] = 0.0f;
        return;
    }
    const uint logitsBase = l.offset + row * l.dimx;
    threadgroup float shared[256];
    const float2 stats = softmaxStats(logits, logitsBase, l.dimx, lane, shared);
    const float scale = indexed ? inverseCount[n.offset] : 1.0f;
    for (uint x = lane; x < l.dimx; x += groupSize)
    {
        const float probability = exp(logits[logitsBase + x] - stats.x) / stats.y;
        const float truth = indexed ? float(x == uint(index)) : target[targetBase + x];
        out[outputBase + x] = (truth - probability) * scale;
    }
}

// Produce a loss and a validity count for each row, keeping logits on the GPU.
kernel void cce_logits_loss(device float* out [[buffer(0)]],
                             device const float* target [[buffer(1)]],
                             device const float* logits [[buffer(2)]],
                             constant MatrixParams& o [[buffer(4)]],
                             constant MatrixParams& t [[buffer(5)]],
                             constant MatrixParams& l [[buffer(6)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint lane [[thread_position_in_threadgroup]])
{
    const uint row = group;
    const uint targetBase = rowBase(t, row, l.dimy);
    const uint logitsBase = l.offset + row * l.dimx;
    threadgroup float shared[256];
    const float2 stats = softmaxStats(logits, logitsBase, l.dimx, lane, shared);
    if (lane != 0) return;
    int index = -1;
    if (t.dimx == 1u) index = int(target[targetBase]);
    else
        for (uint x = 0; x < l.dimx; ++x)
            if (target[targetBase + x] > 0.5f) { index = int(x); break; }
    const bool valid = index >= 0 && uint(index) < l.dimx;
    out[o.offset + row * 2u] = valid ? (stats.x - logits[logitsBase + uint(index)]) + log(stats.y) : 0.0f;
    out[o.offset + row * 2u + 1u] = float(valid);
}

// Only the final scalar is read by the CPU. Index loss is a mean; one-hot loss a sum.
kernel void cce_logits_reduce(device float* out [[buffer(0)]],
                               device const float* rows [[buffer(1)]],
                               constant MatrixParams& o [[buffer(4)]],
                               constant MatrixParams& r [[buffer(5)]],
                               constant MatrixParams& target [[buffer(6)]],
                               uint lane [[thread_position_in_threadgroup]])
{
    threadgroup float2 sums[256];
    float2 sum = float2(0.0f);
    for (uint row = lane; row < r.dimy; row += groupSize)
        sum += float2(rows[r.offset + row * 2u], rows[r.offset + row * 2u + 1u]);
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = groupSize / 2; stride; stride /= 2)
    {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0)
        out[o.offset] = target.dimx == 1u ? (sums[0].y ? sums[0].x / sums[0].y : 0.0f) : sums[0].x;
}
