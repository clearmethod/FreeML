#pragma once

#include <cmath>

#include "ActivationFunction.h"

template<class T>
class Sigmoid : public ActivationFunction<T>
{
public:
    const char* Name()  override { return "sigmoid"; }
    T activate(T x)  override { return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x)); }
    T derivative(T x)  override
    {
        const T s = activate(x);
        return s * (static_cast<T>(1) - s);
    }
};
