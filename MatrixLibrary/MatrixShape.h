#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

enum class MatrixAxes : uint32_t { None=0u, Batch=1u, Channels=2u, Width=4u, Height=8u, All=15u };
enum class MatrixAxis { Batch, Channels, Width, Height };
struct MatrixCoordinate { uint32_t batch=0u,channel=0u,x=0u,y=0u; };
constexpr MatrixAxes operator|(MatrixAxes lhs, MatrixAxes rhs)
{ return static_cast<MatrixAxes>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs)); }

// Width is contiguous: storage[b][c][y][x]. All strides are in elements.
struct MatrixShape
{
    uint32_t batch = 1u;
    uint32_t channels = 1u;
    uint32_t width = 0u;
    uint32_t height = 0u;

    constexpr MatrixShape() = default;
    constexpr MatrixShape(uint32_t b, uint32_t c, uint32_t w, uint32_t h)
        : batch(b), channels(c), width(w), height(h) {}

    static constexpr MatrixShape Matrix(uint32_t columns, uint32_t rows = 1u)
    { return {1u, 1u, columns, rows}; }
    static constexpr MatrixShape Sequence(uint32_t batches, uint32_t features, uint32_t tokens)
    { return {batches, 1u, features, tokens}; }
    static constexpr MatrixShape Image(uint32_t batches, uint32_t channelCount, uint32_t w, uint32_t h)
    { return {batches, channelCount, w, h}; }

    MatrixShape Reduced(MatrixAxes axes) const
    {
        const auto bits = static_cast<uint32_t>(axes);
        if (bits & ~15u) throw std::invalid_argument("Invalid matrix reduction axes.");
        return {(bits&1u) ? 1u : batch, (bits&2u) ? 1u : channels,
                (bits&4u) ? 1u : width, (bits&8u) ? 1u : height};
    }

    MatrixShape BroadcastWith(const MatrixShape& other) const
    {
        auto extent=[](uint32_t a,uint32_t b) {
            if (!a || !b || (a!=b && a!=1u && b!=1u))
                throw std::invalid_argument("Matrix shapes cannot broadcast.");
            return a>b ? a:b;
        };
        return {extent(batch,other.batch),extent(channels,other.channels),
                extent(width,other.width),extent(height,other.height)};
    }

    MatrixShape Concatenated(const MatrixShape& other,MatrixAxis axis) const
    {
        if (axis!=MatrixAxis::Batch && axis!=MatrixAxis::Channels
            && axis!=MatrixAxis::Width && axis!=MatrixAxis::Height)
            throw std::invalid_argument("Invalid concatenation axis.");
        MatrixShape result=*this;
        auto combine=[&](uint32_t& value,uint32_t next,bool selected) {
            if(!selected) {
                if(value!=next) throw std::invalid_argument("Concatenation requires matching non-concatenated axes.");
            } else {
                if(uint64_t(value)+next>UINT32_MAX) throw std::overflow_error("Concatenated axis exceeds indexing limits.");
                value+=next;
            }
        };
        combine(result.batch,other.batch,axis==MatrixAxis::Batch);
        combine(result.channels,other.channels,axis==MatrixAxis::Channels);
        combine(result.width,other.width,axis==MatrixAxis::Width);
        combine(result.height,other.height,axis==MatrixAxis::Height);
        result.CheckedElementCount();return result;
    }

    MatrixAxes AxesReducedTo(const MatrixShape& target) const
    {
        uint32_t bits=0;
        auto axis=[&](uint32_t a,uint32_t b,uint32_t bit) {
            if (a==b) return;
            if (b!=1u) throw std::invalid_argument("Reduction target must retain or reduce each axis.");
            bits|=bit;
        };
        axis(batch,target.batch,1u);axis(channels,target.channels,2u);
        axis(width,target.width,4u);axis(height,target.height,8u);
        return static_cast<MatrixAxes>(bits);
    }

    uint64_t ElementCount() const
    {
        if (!batch || !channels || !width || !height) return 0u;
        uint64_t count = 1u;
        for (uint32_t extent : {batch, channels, height, width})
        {
            if (count > std::numeric_limits<uint64_t>::max() / extent)
                throw std::overflow_error("Matrix element count exceeds 64-bit capacity.");
            count *= extent;
        }
        return count;
    }

    uint32_t CheckedElementCount(uint64_t limit = UINT32_MAX) const
    {
        const uint64_t count = ElementCount();
        if (count > limit || count > UINT32_MAX)
            throw std::overflow_error("Matrix shape exceeds backend allocation/indexing limits.");
        return static_cast<uint32_t>(count);
    }

    size_t ByteSize(size_t elementSize) const
    {
        const uint64_t count = ElementCount();
        if (elementSize && count > std::numeric_limits<size_t>::max() / elementSize)
            throw std::overflow_error("Matrix byte count exceeds addressable capacity.");
        return static_cast<size_t>(count) * elementSize;
    }

    uint64_t Index(uint32_t b, uint32_t c, uint32_t x, uint32_t y) const
    {
        if (b >= batch || c >= channels || x >= width || y >= height)
            throw std::out_of_range("Matrix coordinate is outside its shape.");
        // Verify products before evaluating the addressing expression.
        ElementCount();
        return ((uint64_t(b) * channels + c) * height + y) * width + x;
    }

    bool operator==(const MatrixShape&) const = default;
    bool operator<(const MatrixShape& other) const
    {
        return std::tie(batch, channels, width, height)
            < std::tie(other.batch, other.channels, other.width, other.height);
    }
    std::string GetString() const
    {
        return "(batch=" + std::to_string(batch) + ", channels=" + std::to_string(channels)
            + ", width=" + std::to_string(width) + ", height=" + std::to_string(height) + ")";
    }
};
