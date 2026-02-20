#pragma once

#include <cmath>

#include "ActivationFunction.h"

template<class T>
class Tanh : public ActivationFunction<T>
{
public:
    const char* Name()  override { return "tanh"; }
    T activate(T x)  override { return std::tanh(x); }
    T derivative(T x)  override
    {
        const T t = std::tanh(x);
        return static_cast<T>(1) - t * t;
    }
};
