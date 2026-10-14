#pragma once
#include <cstdint>

template<class T, class MatT>
class LossBase
{
    public:
    virtual ~LossBase() = default;
    virtual const char* GetName() const                                 = 0;
    virtual T       Loss    (MatT* target, MatT* prediction)            = 0;
    virtual void    Gradient(MatT* out, MatT* target, MatT* prediction) = 0;

    // Mean of individual loss evaluations. GPU losses can defer readback until
    // GetAccumulatedLossMean; other losses retain their scalar implementation.
    virtual void ResetAccumulatedLoss()
    {
        m_accumulatedLossSum = 0.0;
        m_accumulatedLossCount = 0u;
    }
    virtual void AccumulateLoss(MatT* target, MatT* prediction)
    {
        m_accumulatedLossSum += Loss(target,prediction);
        ++m_accumulatedLossCount;
    }
    virtual double GetAccumulatedLossMean()
    {
        return m_accumulatedLossCount ? m_accumulatedLossSum / m_accumulatedLossCount : 0.0;
    }
protected:
    double m_accumulatedLossSum = 0.0;
    uint64_t m_accumulatedLossCount = 0u;
};
