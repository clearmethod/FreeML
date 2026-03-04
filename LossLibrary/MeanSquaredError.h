#pragma once

#include <cmath>
#include <cstddef>
#include <LossLibrary/Loss.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#ifdef _WIN32
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
#endif

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
        return MeanSquaredErrorLoss(target, prediction);
    }

    virtual void Gradient(MatT* out, MatT* target, MatT* prediction) override
    {
        Sub(out, target, prediction);
        const size_t count = out->GetElementCount();
        const T invCount = count ? T(1) / static_cast<T>(count) : T(0);
        Scale(out, out, static_cast<float>(T(2) * invCount));
    }
};
