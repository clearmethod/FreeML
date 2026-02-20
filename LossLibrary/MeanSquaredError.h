#pragma once

#include <cmath>
#include <cstddef>
#include <LossLibrary/Loss.h>
#include <MatrixLibrary/MatrixBase.h>

template<class T, class MatT>
class MeanSquaredError : public LossBase<T, MatT>
{
    public:
    const char* GetName() const override
    {
        return "MeanSquaredError";
    }
    virtual T Loss(MatT* target, MatT* prediction) override
    {
        const T* t = target->DataRead();
        const T* p = prediction->DataRead();
        T sum = T(0);
        const size_t count = target->GetElementCount();
        if (count == 0)
        {
            return T(0);
        }

        for(size_t i = 0; i < count; ++i)
        {
            T d = t[i] - p[i];
            sum += d * d;
        }
        return sum / static_cast<T>(count);
    }

    virtual void Gradient(MatT* out, MatT* target, MatT* prediction) override
    {
        Sub(out, target, prediction);
        const size_t count = out->GetElementCount();
        const T invCount = count ? T(1) / static_cast<T>(count) : T(0);
        MapFunction(out, out, [invCount](T& v)
        {
            return v * (T(2) * invCount);
        });
    }
};
