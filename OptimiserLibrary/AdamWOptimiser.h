#pragma once

#include "Optimiser.h"
#include <unordered_map>
#include <vector>
#include <cmath>
#include <type_traits>
#include <MatrixLibrary/MatrixBase_Functions.h>
#ifdef _WIN32
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
#endif

// AdamW optimiser implementation.
// Keeps running estimates of first and second moments for each parameter
// tensor and applies bias corrected updates with decoupled weight decay.
template<class T, class Mat>
class AdamWOptimiser : public Optimiser<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

public:
    struct ParamState
    {
        MatrixRef param;
        MatrixRef moment0;
        MatrixRef moment1;
        uint64_t timestep = 0u;
    };

    AdamWOptimiser(T learningRate = T(0.001),
                   T beta1 = T(0.9),
                   T beta2 = T(0.999),
                   T epsilon = T(1e-8),
                   T weightDecay = T(0.01))
        : m_learningRate(learningRate)
        , m_beta1(beta1)
        , m_beta2(beta2)
        , m_epsilon(epsilon)
        , m_weightDecay(weightDecay)
    {
    }

    virtual std::string GetName() const override
    {
        return "AdamWOptimiser";
	}

    T GetLearningRate() const { return m_learningRate; }
    T GetBeta1() const { return m_beta1; }
    T GetBeta2() const { return m_beta2; }
    T GetEpsilon() const { return m_epsilon; }
    T GetWeightDecay() const { return m_weightDecay; }

    void SetLearningRate(T _in) { m_learningRate = _in; }
    void SetBeta1(T _in)        { m_beta1        = _in; }
    void SetBeta2(T _in)        { m_beta2        = _in; }
    void SetEpsilon(T _in)      { m_epsilon      = _in; }
    void SetWeightDecay(T _in)  { m_weightDecay  = _in; }

    void ExportState(std::vector<ParamState>& out) const
    {
        out.clear();
        out.reserve(m_timestep.size());
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        for (const auto& entry : m_timestep)
        {
            Mat* param = entry.first;
            auto it0 = m_momentzero.find(param);
            auto it1 = m_momentone.find(param);
            if (it0 == m_momentzero.end() || it1 == m_momentone.end())
            {
                continue;
            }
            ParamState state;
            state.param = inst.Acquire(param);
            if (!state.param)
            {
                continue;
            }
            state.moment0 = it0->second;
            state.moment1 = it1->second;
            state.timestep = entry.second;
            out.push_back(state);
        }
    }

    void ClearState()
    {
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        for (auto& entry : m_momentzero)
        {
            inst.RemoveMatrix(entry.second.get());
        }
        for (auto& entry : m_momentone)
        {
            inst.RemoveMatrix(entry.second.get());
        }
        m_momentzero.clear();
        m_momentone.clear();
        m_timestep.clear();
    }

    void ImportState(const std::unordered_map<uint64_t, MatrixRef>& paramsByGuid,
                     const std::vector<ParamState>& in)
    {
        ClearState();
        for (const auto& state : in)
        {
            if (!state.param || !state.moment0 || !state.moment1)
            {
                continue;
            }
            auto it = paramsByGuid.find(state.param->m_guid);
            if (it == paramsByGuid.end() || it->second.get() != state.param.get())
            {
                continue;
            }
            Mat* key = state.param.get();
            m_momentzero[key] = state.moment0;
            m_momentone[key] = state.moment1;
            m_timestep[key] = state.timestep;
        }
    }
    
    // Perform an AdamW optimisation step on the given parameter.
    // Parameter specific state is lazily initialised the first time this is
    // called for a given matrix pointer.
    //
    // param    Parameter matrix to update in place.
    // grad     Gradient of the objective with respect to @p param.
    void Step(Mat* param, Mat* grad) override
    {
        // Use the address of the parameter matrix as a key into the state
        // maps that hold first/second moment estimates and timestep counts.
        Mat* key = param;

        // Lazily create state for unseen parameters.  This ensures that each
        // matrix being optimised gets its own moment buffers and step counter.
        if(m_momentzero.find(key) == m_momentzero.end())
        {
            MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

            m_momentzero[key] = inst.AllocateMatrix(param->GetDims(), "adamw_m");
            m_momentone[key] = inst.AllocateMatrix(param->GetDims(), "adamw_v");
            Clear(m_momentzero[key].get());   // first moment initialised to zero
            Clear(m_momentone[key].get() );    // second moment initialised to zero
            m_timestep[key] = 0;          // timestep starts at zero
        }

        // References to the moment buffers and step count for this parameter
        auto& mt    = m_momentzero[key];
        auto& vt    = m_momentone[key];
        uint64_t& t = m_timestep[key];
        ++t; // increment optimisation step

        // Cache hyperparameters locally for readability
        T lr          = m_learningRate;
        T beta1       = m_beta1;
        T beta2       = m_beta2;
        T eps         = m_epsilon;
        T weightDecay = m_weightDecay;

        // Pre-compute powers used for bias correction
        T beta1_pow_t = std::pow(beta1, t);
        T beta2_pow_t = std::pow(beta2, t);

#ifdef _WIN32
        if constexpr (std::is_same_v<Mat, MatrixDX11<T>>)
        {
            AdamWUpdate(static_cast<MatrixDX11<T>*>(param),
                        static_cast<MatrixDX11<T>*>(grad),
                        static_cast<MatrixDX11<T>*>(mt.get()),
                        static_cast<MatrixDX11<T>*>(vt.get()),
                        lr,
                        beta1,
                        beta2,
                        beta1_pow_t,
                        beta2_pow_t,
                        eps,
                        weightDecay);
        }
        else
#endif
        {
            AdamWUpdate(static_cast<MatrixBase<T>*>(param),
                        static_cast<MatrixBase<T>*>(grad),
                        static_cast<MatrixBase<T>*>(mt.get()),
                        static_cast<MatrixBase<T>*>(vt.get()),
                        lr,
                        beta1,
                        beta2,
                        beta1_pow_t,
                        beta2_pow_t,
                        eps,
                        weightDecay);
        }
    }

private:
    // Per parameter moment estimates and timestep counters
    std::unordered_map<Mat*, MatrixRef> m_momentzero;      // first moment
    std::unordered_map<Mat*, MatrixRef> m_momentone;      // second moment
    std::unordered_map<Mat*, uint64_t>  m_timestep; // step number

    // Hyperparameters controlling the optimisation behaviour
    T m_learningRate;
    T m_beta1;
    T m_beta2;
    T m_epsilon;
    T m_weightDecay;
};
