#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <MatrixLibrary/MatrixBase_Functions.h>

namespace diffusion_example
{
inline constexpr uint32_t kTimeConditionChannels = 4u;

struct NoiseSchedule
{
    std::vector<float> sqrt_alpha_bar;
    std::vector<float> sqrt_one_minus_alpha_bar;
    std::vector<float> sqrt_recip_alpha;
    std::vector<float> coeff;   // (1-alpha) / sqrt_one_minus_alpha_bar
    std::vector<float> sigma;   // sqrt(beta_tilde) posterior sampling stddev
};

inline NoiseSchedule ComputeNoiseSchedule(uint32_t T_steps,
                                          float beta_start = 1e-4f,
                                          float beta_end = 2e-2f)
{
    NoiseSchedule s;
    s.sqrt_alpha_bar.resize(T_steps);
    s.sqrt_one_minus_alpha_bar.resize(T_steps);
    s.sqrt_recip_alpha.resize(T_steps);
    s.coeff.resize(T_steps);
    s.sigma.resize(T_steps);

    float alphaBar = 1.0f;
    for (uint32_t t = 0; t < T_steps; ++t)
    {
        const float beta = beta_start + (beta_end - beta_start) * (static_cast<float>(t) / static_cast<float>(T_steps - 1u));
        const float alpha = 1.0f - beta;
        const float alphaBarPrev = alphaBar;
        alphaBar *= alpha;

        s.sqrt_alpha_bar[t] = std::sqrt(alphaBar);
        s.sqrt_one_minus_alpha_bar[t] = std::sqrt(1.0f - alphaBar);
        s.sqrt_recip_alpha[t] = 1.0f / std::sqrt(alpha);
        s.coeff[t] = beta / s.sqrt_one_minus_alpha_bar[t];

        const float betaTilde = (t == 0u)
            ? 0.0f
            : beta * (1.0f - alphaBarPrev) / (1.0f - alphaBar);
        s.sigma[t] = std::sqrt(std::max(0.0f, betaTilde));
    }
    return s;
}

// Fill a matrix with gaussian noise.
template<class MatType>
inline void SampleGaussian(MatType* mat, std::mt19937& rng)
{
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float* data = mat->DataWrite();
    const size_t n = mat->GetElementCount();
    for (size_t i = 0; i < n; ++i)
    {
        data[i] = dist(rng);
    }
}

// Fill a matrix with the timestep encoded a number of ways to ensure it is
// captured. 
template<class MatType>
inline void FillTimestepConditioning(MatType* mat, uint32_t tIdx, uint32_t totalSteps)
{
    assert(mat);
    assert(mat->GetDimsZ() == kTimeConditionChannels);

    const float denom = (totalSteps > 1u) ? static_cast<float>(totalSteps - 1u) : 1.0f;
    const float tau = static_cast<float>(tIdx) / denom;
    static constexpr float kPi = 3.14159265358979323846f;

    const float values[kTimeConditionChannels] =
    {
        tau,
        tau * tau,
        std::sin(kPi * tau),
        std::cos(kPi * tau)
    };

    for (uint32_t z = 0; z < kTimeConditionChannels; ++z)
    {
        MatType sliceView;
        mat->GetSliceZ(&sliceView, z);
        Fill(&sliceView, values[z]);
    }
}

// Generates a checkerboard pattern as training data.
template<class MatType>
inline void GenerateCheckerboard(MatType* mat, uint32_t tileSize)
{
    const uint32_t width = mat->GetDimsX();
    const uint32_t height = mat->GetDimsY();
    const uint32_t depth = mat->GetDimsZ();
    for (uint32_t z = 0; z < depth; ++z)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                const bool light = (((x / tileSize) + (y / tileSize)) % 2u) == 0u;
                mat->SetValue(x, y, z, light ? 1.0f : -1.0f);  // [-1, 1] normalised
            }
        }
    }
}

template<class T>
inline void ApplyForwardNoise(T* zOut,
                              const T* z0,
                              const T* eps,
                              size_t count,
                              const NoiseSchedule& sched,
                              uint32_t tIdx)
{
    const float sab = sched.sqrt_alpha_bar[tIdx];
    const float s1ab = sched.sqrt_one_minus_alpha_bar[tIdx];
    for (size_t i = 0; i < count; ++i)
    {
        zOut[i] = static_cast<T>(sab * z0[i] + s1ab * eps[i]);
    }
}

template<class T>
inline void DDPMStep(T* zOut,
                     const T* zt,
                     const T* predEps,
                     size_t count,
                     const NoiseSchedule& sched,
                     uint32_t tIdx,
                     std::mt19937& rng)
{
    const float recipAlpha = sched.sqrt_recip_alpha[tIdx];
    const float coeff = sched.coeff[tIdx];
    const float sigma = sched.sigma[tIdx];
    std::normal_distribution<float> noiseDist(0.0f, 1.0f);
    for (size_t i = 0; i < count; ++i)
    {
        const float z = (tIdx > 0u) ? noiseDist(rng) : 0.0f;
        zOut[i] = static_cast<T>(recipAlpha * (zt[i] - coeff * predEps[i]) + sigma * z);
    }
}

// Cosine noise schedule (Nichol & Dhariwal, "Improved DDPM", 2021).
// Distributes noise uniformly — fixes linear schedule's near-zero noise at t=0
// that wastes training on easy low-noise timesteps.
inline NoiseSchedule ComputeCosineNoiseSchedule(uint32_t T_steps, float s = 0.008f)
{
    NoiseSchedule sched;
    sched.sqrt_alpha_bar.resize(T_steps);
    sched.sqrt_one_minus_alpha_bar.resize(T_steps);
    sched.sqrt_recip_alpha.resize(T_steps);
    sched.coeff.resize(T_steps);
    sched.sigma.resize(T_steps);

    static constexpr float kPi = 3.14159265358979323846f;
    auto f = [&](float t) -> float {
        const float arg = (t / static_cast<float>(T_steps) + s) / (1.0f + s) * (kPi * 0.5f);
        return std::cos(arg) * std::cos(arg);
    };
    const float f0 = f(0.0f);
    float alphaBarPrev = 1.0f;
    for (uint32_t t = 0; t < T_steps; ++t)
    {
        const float alphaBar = f(static_cast<float>(t + 1u)) / f0;
        const float beta     = std::min(1.0f - alphaBar / alphaBarPrev, 0.999f);
        const float alpha    = 1.0f - beta;
        sched.sqrt_alpha_bar[t]           = std::sqrt(std::max(0.0f, alphaBar));
        sched.sqrt_one_minus_alpha_bar[t] = std::sqrt(std::max(0.0f, 1.0f - alphaBar));
        sched.sqrt_recip_alpha[t]         = 1.0f / std::sqrt(alpha);
        sched.coeff[t]                    = beta / std::max(sched.sqrt_one_minus_alpha_bar[t], 1e-6f);
        const float betaTilde             = (t == 0u)
            ? 0.0f
            : beta * (1.0f - alphaBarPrev) / std::max(1.0f - alphaBar, 1e-8f);
        sched.sigma[t]                    = std::sqrt(std::max(0.0f, betaTilde));
        alphaBarPrev                      = alphaBar;
    }
    return sched;
}

// Deterministic DDIM sampling step (Song et al. 2021).
// Drop-in for DDPMStep at inference; same eps-prediction training objective.
// With beta_end=0.5 or cosine schedule, DDPM sigma[9]≈0.69 dominates each step —
// DDIM eliminates this, giving clean deterministic denoising.
template<class T>
inline void TimeStepSample(T* zOut,
                     const T* zt,
                     const T* predEps,
                     size_t count,
                     const NoiseSchedule& sched,
                     uint32_t tIdx)
{
    const float sqrt_ab_t     = sched.sqrt_alpha_bar[tIdx];
    const float sqrt_1ab_t    = sched.sqrt_one_minus_alpha_bar[tIdx];
    const float sqrt_ab_prev  = (tIdx > 0u) ? sched.sqrt_alpha_bar[tIdx - 1u]             : 1.0f;
    const float sqrt_1ab_prev = (tIdx > 0u) ? sched.sqrt_one_minus_alpha_bar[tIdx - 1u]   : 0.0f;
    // Guard against alpha_bar ≈ 0 at the final cosine-schedule step.
    const float inv_sqrt_ab   = (sqrt_ab_t > 1e-4f) ? (1.0f / sqrt_ab_t) : 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        const float z   = static_cast<float>(zt[i]);
        const float eps = static_cast<float>(predEps[i]);
        // Estimate clean latent x0, then re-noise deterministically for t-1.
        const float x0  = (z - sqrt_1ab_t * eps) * inv_sqrt_ab;
        zOut[i] = static_cast<T>(sqrt_ab_prev * x0 + sqrt_1ab_prev * eps);
    }
}
}
