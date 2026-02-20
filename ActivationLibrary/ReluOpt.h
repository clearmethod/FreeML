#pragma once

#include <cmath>

#include "ActivationFunction.h"

template<class T>
class ReluOpt : public ActivationFunction<T>
{
public:
    const char* Name()  override { return "relu_opt"; }
    T activate(T x)  override { return static_cast<T>(0.5) * (x + std::fabs(x)); }
    T derivative(T x)  override { return static_cast<T>(x > static_cast<T>(0)); }
};
