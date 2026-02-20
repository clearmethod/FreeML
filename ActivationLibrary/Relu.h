#pragma once

#include <algorithm>

#include "ActivationFunction.h"

template<class T>
class Relu : public ActivationFunction<T>
{
public:
    const char* Name()  override { return "relu"; }
    T activate(T x)  override { return std::max(T(0), x); }
    T derivative(T x)  override { return x > T(0) ? T(1) : T(0); }
};
