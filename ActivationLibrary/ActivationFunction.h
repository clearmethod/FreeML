#pragma once

#include <MatrixLibrary/MatrixBase.h>
#include <MatrixLibrary/MatrixBase_Functions.h>

// Base class for activations operating over matrices and scalars
template<class T, class Mat=MatrixBase<T>>
class ActivationFunction
{
public:
    virtual ~ActivationFunction() = default;

    virtual const char* Name() = 0;
    virtual T activate(T x) = 0;
    virtual T derivative(T x) = 0;

    virtual void activateMat(Mat* _matOut, Mat* _matIn)
    {
        MapFunction(_matOut, _matIn, [this](T value) { return activate(value); });
    }

    virtual void activateMat(Mat* _matOut,
                             Mat* _matIn,
                             ThreadPool* _pool)
    {
        if (_pool == nullptr)
        {
            activateMat(_matOut, _matIn);
            return;
        }

        MapFunction(_matOut,
                    static_cast<Mat*>(_matIn),
                    [this](T value) { return activate(value); },
                    _pool);
    }

    virtual void derivativeMat(Mat* _matOut,
                               Mat* _matIn,
                               Mat* _notused = nullptr) 
    {
        (void)_notused;
        MapFunction(_matOut, _matIn, [this](T value) { return derivative(value); });
    }
};
