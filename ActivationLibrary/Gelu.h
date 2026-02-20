#pragma once

#include <cmath>

#include "ActivationFunction.h"
#ifdef _WIN32
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
#endif

template<class T, class Mat>
class Gelu : public ActivationFunction<T, Mat>
{
public:
    using ActivationFunction<T, Mat>::activateMat;
    using ActivationFunction<T, Mat>::derivativeMat;

    const char* Name() override { return "gelu"; }
    T activate(T x) override
    {
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + std::erf(x / std::sqrt(static_cast<T>(2))));
    }

    T derivative(T x) override
    {
        const T sqrt2 = std::sqrt(static_cast<T>(2));
        const T sqrt2pi = std::sqrt(static_cast<T>(2) * static_cast<T>(3.14159265358979323846));
        const T erfTerm = std::erf(x / sqrt2);
        const T expTerm = std::exp(static_cast<T>(-0.5) * x * x);
        return static_cast<T>(0.5) * (static_cast<T>(1) + erfTerm) + (x * expTerm) / sqrt2pi;
    }

    virtual void activateMat(Mat* _matOut,
                             Mat* _matIn,
                             ThreadPool* _pool)
    {
        (void)_pool;
        GeluMat(_matOut, _matIn);
    }

    virtual void derivativeMat(Mat* _matOut,
                               Mat* _matIn,
                               Mat* _notused = nullptr) 
    {
        (void)_notused;
        GeluDerivtiveMat(_matOut, _matIn);
    }
};
