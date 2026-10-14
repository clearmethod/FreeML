#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>

// Progress is measured in completed optimizer updates, not sequence passes.
struct NanoLLMLearningRateSchedule
{
    float peak;
    float minimum;
    uint64_t warmupUpdates;
    uint64_t decayUpdates;

    bool IsValid() const
    {
        return std::isfinite(peak) && std::isfinite(minimum)
            && minimum > 0.0f && minimum <= peak
            && decayUpdates > warmupUpdates;
    }

    float AtUpdate(uint64_t update) const
    {
        assert(IsValid());
        if (update < warmupUpdates)
            return float(double(peak) * (double(update) + 1.0)
                         / (double(warmupUpdates) + 1.0));
        if (update >= decayUpdates)
            return minimum;

        constexpr double pi = 3.14159265358979323846;
        const double progress = double(update - warmupUpdates)
                              / double(decayUpdates - warmupUpdates);
        const double weight = 0.5 * (1.0 + std::cos(pi * progress));
        return float(minimum + weight * (double(peak) - minimum));
    }
};
