#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <LossLibrary/Loss.h>
#include <MatrixLibrary/MatrixBase.h>

template<class T, class MatT>
class BinaryCrossEntropy : public LossBase<T, MatT>
{
    public:
    const char* GetName() const override
    {
        return "BinaryCrossEntropy";
    }

    virtual T Loss(MatT* target, MatT* prediction) override
    {
        constexpr T epsilon = T(1e-7);
        T loss = T(0);
        const size_t count = target->GetElementCount();
        const T* t = target->DataRead();
        const T* p = prediction->DataRead();
        for(size_t i = 0; i < count; ++i)
        {
            T pc = std::clamp(p[i], epsilon, T(1) - epsilon);
            loss += -(t[i] * std::log(pc) + (T(1) - t[i]) * std::log(T(1) - pc));
        }
        return loss / T(count);
    }

    virtual void Gradient(MatT* out, MatT* target, MatT* prediction) override
    {
        constexpr T epsilon = T(1e-7);
        const size_t count = out->GetElementCount();
        const T inv_n = T(1) / T(count);
        const T* t = target->DataRead();
        const T* p = prediction->DataRead();
        T* o = out->DataWrite();
        for(size_t i = 0; i < count; ++i)
        {
            T pc = std::clamp(p[i], epsilon, T(1) - epsilon);
            T grad = (pc - t[i]) / (pc * (T(1) - pc));
            grad = -(grad * inv_n);
            o[i] = grad;
        }
    }
};
