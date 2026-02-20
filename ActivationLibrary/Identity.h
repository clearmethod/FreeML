#pragma once

#include "ActivationFunction.h"

template<class T>
class Identity : public ActivationFunction<T>
{
public:
    const char* Name() override { return "identity"; }
    T activate(T x) override { return x; }
    T derivative(T) override { return static_cast<T>(1); }
};
