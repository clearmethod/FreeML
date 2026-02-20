#pragma once

#include <random>
#include <atomic>
#include <cstddef>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>


namespace RandomUtils 
{
    // Random number generator setup (shared across calls)
    inline std::mt19937& GetGenerator() 
    {
        static std::mt19937 generator(std::random_device{}());
        return generator;
    }

    // Random float from normal distribution
    inline float random_normal(float mean = 0.0f, float stddev = 1.0f) 
    {
        static thread_local std::normal_distribution<float> dist;
        dist = std::normal_distribution<float>(mean, stddev);
        return dist(GetGenerator());
    }

    // Optional: random float from uniform distribution
    inline float random_uniform(float min = 0.0f, float max = 1.0f) 
    {
        static thread_local std::uniform_real_distribution<float> dist;
        dist = std::uniform_real_distribution<float>(min, max);
        return dist(GetGenerator());
    }

    inline std::vector<float>& GetUniformBuffer()
    {
        static std::vector<float> buffer = []()
        {
            constexpr size_t kBufferBytes = 16u * 1024u * 1024u;
            constexpr size_t kCount = kBufferBytes / sizeof(float);
            std::vector<float> data(kCount);
            for (size_t i = 0; i < kCount; ++i)
            {
                data[i] = random_uniform(0.0f, 1.0f);
            }
            return data;
        }();
        return buffer;
    }

    inline size_t GetUniformBufferCount()
    {
        return GetUniformBuffer().size();
    }

    inline const float* GetUniformBufferData()
    {
        return GetUniformBuffer().data();
    }

    inline float random_uniform_buffered()
    {
        static std::atomic<size_t> index{0u};
        auto& buffer = GetUniformBuffer();
        const size_t i = index.fetch_add(1u, std::memory_order_relaxed);
        return buffer[i % buffer.size()];
    }
}

inline void PressAnyKeyToContinue()
{
    std::cout << "Press Enter to continue . . .";
    std::cin.get();
}

static uint64_t GenerateGuidUINT()
{
    static thread_local std::mt19937_64 rng{ std::random_device{}() };
    std::uniform_int_distribution<uint64_t> dist;
    const uint64_t a = dist(rng);
    return a;
}

static std::string GenerateGuid()
{
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    const uint64_t a = dist(rng);
    const uint64_t b = dist(rng);

    std::ostringstream ss;
    ss << std::hex << std::setfill('0')
        << std::setw(16) << a
        << std::setw(16) << b;
    return ss.str();
}
