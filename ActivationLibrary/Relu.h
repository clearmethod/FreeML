#pragma once

#include <algorithm>

#include "ActivationFunction.h"
#ifdef _WIN32
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
#endif

template<class T>
class Relu : public ActivationFunction<T>
{
public:
    using ActivationFunction<T>::activateMat;
    using ActivationFunction<T>::derivativeMat;

    const char* Name()  override { return "relu"; }
    T activate(T x)  override { return std::max(T(0), x); }
    T derivative(T x)  override { return x > T(0) ? T(1) : T(0); }

    template<class Mat>
    void activateMat(Mat* _matOut, Mat* _matIn)
    {
        ReluMat(_matOut, _matIn);
    }

    template<class Mat>
    void activateMat(Mat* _matOut,
                     Mat* _matIn,
                     ThreadPool* _pool)
    {
        (void)_pool;
        ReluMat(_matOut, _matIn);
    }

    template<class Mat>
    void derivativeMat(Mat* _matOut,
                       Mat* _matIn,
                       Mat* _notused = nullptr)
    {
        (void)_notused;
        ReluDerivativeMat(_matOut, _matIn);
    }
};
