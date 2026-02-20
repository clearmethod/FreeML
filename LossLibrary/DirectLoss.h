#pragma once

#include <cmath>
#include <cstddef>

#include <LossLibrary/Loss.h>

#include <MatrixLibrary/MatrixBase.h>

template<class T, class MatT>
class DirectLoss : public LossBase<T,MatT>
{
    public:
    const char* GetName() const override
    {
        return "DirectLoss";
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
        return sum;
    }

    virtual void Gradient(MatT* out, MatT* target, MatT* prediction) override
    {
        Sub(out, target, prediction);
    }
};
