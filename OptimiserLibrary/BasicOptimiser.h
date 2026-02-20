#pragma once

#include "Optimiser.h"
#include <unordered_map>
#include <cmath>

template<class T, class Mat>
class BasicOptimiser : public Optimiser<T, Mat>
{
public:
    BasicOptimiser(T learningRate = T(0.001))
        : m_learningRate(learningRate)
    {}

    virtual std::string GetName() const override
    {
        return "BasicOptimiser";
	}

    T GetLearningRate() const
    {
        return static_cast<T>(m_learningRate);
    }

    // param    Parameter matrix to update in place.
    // grad     Gradient of the objective with respect to @p param.
    void Step(Mat* param, Mat* grad) override
    {
        assert(param->GetElementCount() == grad->GetElementCount());

        // Scale and add the gradient.
        double lr = m_learningRate;
        PerElement_Func(param, param, grad, [lr](T& _L, T& _R) -> T {return _L + (_R * lr); });
    }

private:
    // Hyperparameters controlling the optimisation behaviour
    double m_learningRate;
};

