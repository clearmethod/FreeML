#pragma once

#include <cmath>
#include <cstddef>
#include <LossLibrary/Loss.h>
#include <MatrixLibrary/MatrixBase.h>

template<class T, class MatT>
class DirectNormalisedLoss : public LossBase<T, MatT>
{
    public:
    const char* GetName() const override
    {
        return "DirectNormalisedLoss";
    }
    
    virtual T Loss(MatT* target, MatT* prediction) override
    {
        const T* t = target->DataRead();
        const T* p = prediction->DataRead();
        T sum = T(0);
        const size_t count = target->GetElementCount();
        for(size_t i = 0; i < count; ++i)
        {
            sum += std::abs(t[i] - p[i]);
        }
        return sum / T(count);
    }

    virtual void Gradient(MatT* out, MatT* target, MatT* prediction) override
    {
        Sub(out, target, prediction);
        T sum = T(0);
        const size_t count = out->GetElementCount();
        const T* data = out->DataRead();
        for(size_t i = 0; i < count; ++i)
        {
            sum += std::abs(data[i]);
        }
        MapFunction(out, out, [sum](T& v){ return v / sum; });
    }
};
