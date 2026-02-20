#pragma once

#include "../MatrixLibrary/MatrixBase.h"

#include <cstdint>

/**
 * Base class for optimisation algorithms.
 */
template<class T, class Mat>
class Optimiser
{
public:
    virtual ~Optimiser() = default;

    virtual std::string GetName() const
    {
        return "Optimiser Base";
	}
    /**
     * Update a parameter tensor in place.
     *
     * param    Parameter matrix to be updated.
     * grad     Gradient matrix computed for @p param.
     * layerId  Optional identifier for grouping parameters.
     */
    virtual void Step(Mat* param, Mat* grad) = 0;
};
