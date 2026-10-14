#pragma once

#include <MatrixLibrary/MatrixManager.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Reused GPU reduction storage. Each partial holds (max absolute value,
// sum of squares normalized by that maximum), avoiding float square overflow.
template<class T, class Mat>
struct GradientClipWorkspace
{
    using MatrixRef = typename MatrixManager<T,Mat>::MatrixRef;
    static constexpr uint32_t elementsPerPartial = 1024u;
    MatrixRef partials;
    MatrixRef scale;

    uint32_t Prepare(const std::vector<Mat*>& gradients)
    {
        uint64_t count = 0;
        for (auto* gradient : gradients)
        {
            if (!gradient) throw std::invalid_argument("Null gradient in norm clipping.");
            count += (uint64_t(gradient->GetElementCount()) + elementsPerPartial - 1u) / elementsPerPartial;
        }
        if (count > std::numeric_limits<uint32_t>::max() / 2u)
            throw std::length_error("Too many gradient reduction partials.");
        if (count == 0) return 0;
        auto& manager = MatrixManager<T,Mat>::Instance();
        if (!partials || partials->GetDimsY() < count)
            partials = manager.AllocateMatrix({2u,uint32_t(count),1u},"GradientNormPartials");
        if (!scale)
            scale = manager.AllocateMatrix({1u,1u,1u},"GradientClipScale");
        return uint32_t(count);
    }

    void Reset() { partials = {}; scale = {}; }
};
