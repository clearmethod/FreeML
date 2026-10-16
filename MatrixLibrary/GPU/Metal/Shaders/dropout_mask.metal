#include <metal_stdlib>
using namespace metal;

struct MatrixParams
{
    uint dimx, dimy, dimz, offset;
    uint uniqueId, channels, batch, reserved;
    uint optionalParams[16];
};

uint pcg_hash(uint value)
{
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

kernel void dropout_mask(device float* out [[buffer(0)]],
                         constant MatrixParams& params [[buffer(1)]],
                         uint idx [[thread_position_in_grid]])
{
    if (idx >= params.dimx * params.dimy * params.dimz) return;
    const float probability = as_type<float>(params.optionalParams[0]);
    const uint stream = pcg_hash(params.optionalParams[1]);
    // Keep the upper 24 bits: conversion is exact and cannot round up to 1.
    const float uniform = float(pcg_hash(idx ^ stream) >> 8u) * (1.0f / 16777216.0f);
    out[params.offset + idx] = uniform < probability ? 0.0f : 1.0f;
}
