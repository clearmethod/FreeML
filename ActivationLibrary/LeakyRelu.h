#pragma once

#include "ActivationFunction.h"

// LeakyReLU activation keeps a small gradient for negative inputs to avoid dead neurons.
template<class T>
class LeakyRelu : public ActivationFunction<T>
{
public:
    explicit LeakyRelu(T negativeSlope = T(0.01))
        : m_negativeSlope(negativeSlope)
    {}

    const char* Name() override { return "leaky_relu"; }

    T activate(T x) override
    {
        return x > T(0) ? x : m_negativeSlope * x;
    }

    T derivative(T x) override
    {
        return x > T(0) ? T(1) : m_negativeSlope;
    }

private:
    T m_negativeSlope;
};
