#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <LossLibrary/Loss.h>
#include <MatrixLibrary/MatrixBase.h>

#ifdef _WIN32
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11.h>
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
#include <MatrixLibrary/MatrixManager.h>
#endif

// Multi-class categorical cross-entropy with logits.
// Supports either one-hot targets (vocab x T) or index targets (1 x T).
// When using index targets, entries equal to -1 are ignored (ignore_index).
// Gradient is written with respect to the logits (pre-activation), scaled as negative gradient
// to match the optimiser's `param += lr * grad` convention elsewhere in the codebase.
template<class T, class MatT>
class CategoricalCrossEntropyWithLogits : public LossBase<T, MatT>
{
    public:
    const char* GetName() const override
    {
        return "CategoricalCrossEntropyWithLogits";
    }

    virtual T Loss(MatT* target, MatT* logits) override
    {
        const uint32_t vocab = logits->GetDimsX();
        const uint32_t steps = logits->GetDimsY();
        const uint32_t batch = logits->GetDimsZ();
        if (vocab == 0 || steps == 0 || batch == 0) return T(0);

        if (target->GetDimsX() == 1u)
        {
            uint32_t valid = 0u;
            T total = T(0);
            const uint32_t targetBatch = target->GetDimsZ();
            for (uint32_t z = 0; z < batch; ++z)
            {
                const uint32_t tz = targetBatch == 1u ? 0u : z;
                for (uint32_t t = 0; t < steps; ++t)
                {
                    const int32_t idx = static_cast<int32_t>(target->GetValue(0, t, tz));
                    if (idx == -1)
                    {
                        continue;
                    }
                    assert(idx >= 0 && static_cast<uint32_t>(idx) < vocab);

                    // Find max logit for numerical stability
                    T maxLogit = logits->GetValue(0, t, z);
                    for (uint32_t i = 1; i < vocab; ++i)
                    {
                        T logit = logits->GetValue(i, t, z);
                        if (logit > maxLogit) maxLogit = logit;
                    }

                    // Compute log-sum-exp
                    T sumExp = T(0);
                    for (uint32_t i = 0; i < vocab; ++i)
                    {
                        sumExp += std::exp(logits->GetValue(i, t, z) - maxLogit);
                    }
                    const T logSumExp = maxLogit + std::log(sumExp);

                    total += (logSumExp - logits->GetValue(static_cast<uint32_t>(idx), t, z));
                    ++valid;
                }
            }

            if (valid == 0u)
            {
                return T(0);
            }
            return total / static_cast<T>(valid);
        }

        T total = T(0);
        const uint32_t targetBatch = target->GetDimsZ();
        for (uint32_t z = 0; z < batch; ++z)
        {
            const uint32_t tz = targetBatch == 1u ? 0u : z;
            for (uint32_t t = 0; t < steps; ++t)
            {
                // Find max logit for numerical stability
                T maxLogit = logits->GetValue(0, t, z);
                for (uint32_t i = 1; i < vocab; ++i)
                {
                    T logit = logits->GetValue(i, t, z);
                    if (logit > maxLogit) maxLogit = logit;
                }

                // Compute log-sum-exp
                T sumExp = T(0);
                for (uint32_t i = 0; i < vocab; ++i)
                {
                    sumExp += std::exp(logits->GetValue(i, t, z) - maxLogit);
                }
                const T logSumExp = maxLogit + std::log(sumExp);

                // Cross-entropy for one-hot target: -log p_y = -(z_y - logSumExp)
                for (uint32_t i = 0; i < vocab; ++i)
                {
                    const T ti = target->GetValue(i, t, tz);
                    if (ti > T(0.5))
                    {
                        total += (logSumExp - logits->GetValue(i, t, z));
                        break;
                    }
                }
            }
        }
        return total;
    }

    virtual void Gradient(MatT* out, MatT* target, MatT* logits) override
    {
        const uint32_t vocab = logits->GetDimsX();
        const uint32_t steps = logits->GetDimsY();
        const uint32_t batch = logits->GetDimsZ();
        if (vocab == 0 || steps == 0 || batch == 0) return;

        if (target->GetDimsX() == 1u)
        {
            uint32_t valid = 0u;
            const uint32_t targetBatch = target->GetDimsZ();
            for (uint32_t z = 0; z < batch; ++z)
            {
                const uint32_t tz = targetBatch == 1u ? 0u : z;
                for (uint32_t t = 0; t < steps; ++t)
                {
                    if (static_cast<int32_t>(target->GetValue(0, t, tz)) != -1)
                    {
                        ++valid;
                    }
                }
            }

            const T invValid = valid > 0u ? (T(1) / static_cast<T>(valid)) : T(0);
            for (uint32_t z = 0; z < batch; ++z)
            {
                const uint32_t tz = targetBatch == 1u ? 0u : z;
                for (uint32_t t = 0; t < steps; ++t)
                {
                    const int32_t idx = static_cast<int32_t>(target->GetValue(0, t, tz));
                    if (idx == -1)
                    {
                        for (uint32_t i = 0; i < vocab; ++i)
                        {
                            out->SetValue(i, t, z, T(0));
                        }
                        continue;
                    }
                    assert(idx >= 0 && static_cast<uint32_t>(idx) < vocab);

                    // Stable softmax: compute probabilities
                    T maxLogit = logits->GetValue(0, t, z);
                    for (uint32_t i = 1; i < vocab; ++i)
                    {
                        T logit = logits->GetValue(i, t, z);
                        if (logit > maxLogit) maxLogit = logit;
                    }

                    T sumExp = T(0);
                    for (uint32_t i = 0; i < vocab; ++i)
                    {
                        sumExp += std::exp(logits->GetValue(i, t, z) - maxLogit);
                    }

                    for (uint32_t i = 0; i < vocab; ++i)
                    {
                        const T p = std::exp(logits->GetValue(i, t, z) - maxLogit) / sumExp;
                        const T ti = (static_cast<uint32_t>(idx) == i) ? T(1) : T(0);
                        out->SetValue(i, t, z, (ti - p) * invValid);
                    }
                }
            }
            return;
        }

        const uint32_t targetBatch = target->GetDimsZ();
        for (uint32_t z = 0; z < batch; ++z)
        {
            const uint32_t tz = targetBatch == 1u ? 0u : z;
            for (uint32_t t = 0; t < steps; ++t)
            {
                // Stable softmax: compute probabilities
                T maxLogit = logits->GetValue(0, t, z);
                for (uint32_t i = 1; i < vocab; ++i)
                {
                    T logit = logits->GetValue(i, t, z);
                    if (logit > maxLogit) maxLogit = logit;
                }

                T sumExp = T(0);
                for (uint32_t i = 0; i < vocab; ++i)
                {
                    sumExp += std::exp(logits->GetValue(i, t, z) - maxLogit);
                }

                for (uint32_t i = 0; i < vocab; ++i)
                {
                    const T p  = std::exp(logits->GetValue(i, t, z) - maxLogit) / sumExp;
                    const T ti = target->GetValue(i, t, tz);
                    // Negative gradient wrt logits: (t - p). No extra averaging here.
                    out->SetValue(i, t, z, (ti - p));
                }
            }
        }
    }
};

#ifdef _WIN32
template<class T>
class  CategoricalCrossEntropyWithLogits<T, MatrixDX11<T>> : public LossBase<T, MatrixDX11<T>>
{
    public:
    const char* GetName() const override
    {
        return "CategoricalCrossEntropyWithLogits";
    }

    virtual T Loss(MatrixDX11<T>* target, MatrixDX11<T>* logits) override
    {
        if (!target || !logits)
        {
            return T(0);
        }

        const uint32_t vocab = logits->GetDimsX();
        const uint32_t steps = logits->GetDimsY();
        const uint32_t batch = logits->GetDimsZ();
        if (vocab == 0 || steps == 0 || batch == 0)
        {
            return T(0);
        }

        const uint32_t rows = steps * batch;
        if (rows == 0u)
        {
            return T(0);
        }

        MatrixManager<T, MatrixDX11<T>>& inst = MatrixManager<T, MatrixDX11<T>>::Instance();
        MatrixDX11<T>* scratch = inst.GetScratch({ 2u, rows, 1u });
        if (!scratch)
        {
            return T(0);
        }

        target->Unmap();
        logits->Unmap();
        scratch->Unmap();

        DirectX11Manager* manager = logits->m_gInstance;
        ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
        ID3D11ComputeShader* shader = manager ? manager->GetShader("cce_logits_loss") : nullptr;
        if (!context || !shader)
        {
            return T(0);
        }

        if (!SyncCBuffer(scratch) || !SyncCBuffer(target) || !SyncCBuffer(logits))
        {
            return T(0);
        }

        ID3D11Buffer* outCBuffer = manager->GetCBuffer(scratch->m_dataHandles.m_cbufferHandle);
        ID3D11Buffer* targetCBuffer = manager->GetCBuffer(target->m_dataHandles.m_cbufferHandle);
        ID3D11Buffer* logitsCBuffer = manager->GetCBuffer(logits->m_dataHandles.m_cbufferHandle);
        if (!outCBuffer || !targetCBuffer || !logitsCBuffer)
        {
            return T(0);
        }

        ID3D11ShaderResourceView* targetSRV = manager->GetBufferSRV(target->m_dataHandles.m_bufferHandle);
        ID3D11ShaderResourceView* logitsSRV = manager->GetBufferSRV(logits->m_dataHandles.m_bufferHandle);
        ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(scratch->m_dataHandles.m_bufferHandle);
        if (!targetSRV || !logitsSRV || !outUAV)
        {
            return T(0);
        }

        context->CSSetShader(shader, nullptr, 0u);
        ID3D11ShaderResourceView* srvs[] = { targetSRV, logitsSRV };
        context->CSSetShaderResources(1u, 2u, srvs);
        ID3D11UnorderedAccessView* uavs[] = { outUAV };
        context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
        ID3D11Buffer* cbs[] = { outCBuffer, targetCBuffer, logitsCBuffer };
        context->CSSetConstantBuffers(0u, 3u, cbs);

        context->Dispatch(rows, 1u, 1u);

        ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
        context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
        context->CSSetShaderResources(1u, 2u, nullSRVs);
        context->CSSetShader(nullptr, nullptr, 0u);

        const T* lossPtr = scratch->DataRead();
        if (!lossPtr)
        {
            return T(0);
        }

        T total = T(0);
        if (target->GetDimsX() == 1u)
        {
            T valid = T(0);
            for (uint32_t row = 0; row < rows; ++row)
            {
                total += lossPtr[row * 2u];
                valid += lossPtr[row * 2u + 1u];
            }
            scratch->Unmap();
            if (valid == T(0))
            {
                return T(0);
            }
            return total / valid;
        }

        for (uint32_t row = 0; row < rows; ++row)
        {
            total += lossPtr[row * 2u];
        }
        scratch->Unmap();
        if (GPUSyncCalls)
            DirectX11Manager::Instance()->WaitForGPU();

        return total;

    }

    virtual void Gradient(MatrixDX11<T>* out, MatrixDX11<T>* target, MatrixDX11<T>* logits) override
    {
        if (!out || !target || !logits)
        {
            return;
        }

        const uint32_t vocab = logits->GetDimsX();
        const uint32_t steps = logits->GetDimsY();
        const uint32_t batch = logits->GetDimsZ();
        if (vocab == 0 || steps == 0 || batch == 0)
        {
            return;
        }

        if (out->GetDimsX() != vocab || out->GetDimsY() != steps || out->GetDimsZ() != batch)
        {
            return;
        }

        T invValid = T(0);
        if (target->GetDimsX() == 1u)
        {
            uint32_t valid = 0u;
            const uint32_t targetBatch = target->GetDimsZ();
            const uint32_t targetPlane = target->GetDimsX() * target->GetDimsY();
            const T* targetPtr = target->DataRead();
            if (targetPtr)
            {
                for (uint32_t z = 0; z < batch; ++z)
                {
                    const uint32_t tz = targetBatch == 1u ? 0u : z;
                    const uint32_t targetBase = tz * targetPlane;
                    for (uint32_t t = 0; t < steps; ++t)
                    {
                        const int32_t idx = static_cast<int32_t>(targetPtr[targetBase + t]);
                        if (idx != -1)
                        {
                            ++valid;
                        }
                    }
                }
            }
            target->Unmap();
            invValid = valid > 0u ? (T(1) / static_cast<T>(valid)) : T(0);
        }

        out->Unmap();
        target->Unmap();
        logits->Unmap();

        DirectX11Manager* manager = logits->m_gInstance;
        ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
        ID3D11ComputeShader* shader = manager ? manager->GetShader("cce_logits_grad") : nullptr;
        if (!context || !shader)
        {
            return;
        }

        if (!SyncCBuffer(out) || !SyncCBuffer(target) || !SyncCBuffer(logits))
        {
            return;
        }

        if (target->GetDimsX() == 1u)
        {
            const uint32_t optionalParams[1] = { PackFloat(invValid) };
            if (!manager->SetCBufferOptionalParams(out->m_dataHandles.m_cbufferHandle, optionalParams, 1u))
            {
                return;
            }
        }
        else
        {
            const uint32_t optionalParams[1] = { PackFloat(T(0)) };
            manager->SetCBufferOptionalParams(out->m_dataHandles.m_cbufferHandle, optionalParams, 1u);
        }

        ID3D11Buffer* outCBuffer = manager->GetCBuffer(out->m_dataHandles.m_cbufferHandle);
        ID3D11Buffer* targetCBuffer = manager->GetCBuffer(target->m_dataHandles.m_cbufferHandle);
        ID3D11Buffer* logitsCBuffer = manager->GetCBuffer(logits->m_dataHandles.m_cbufferHandle);
        if (!outCBuffer || !targetCBuffer || !logitsCBuffer)
        {
            return;
        }

        ID3D11ShaderResourceView* targetSRV = manager->GetBufferSRV(target->m_dataHandles.m_bufferHandle);
        ID3D11ShaderResourceView* logitsSRV = manager->GetBufferSRV(logits->m_dataHandles.m_bufferHandle);
        ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(out->m_dataHandles.m_bufferHandle);
        if (!targetSRV || !logitsSRV || !outUAV)
        {
            return;
        }

        context->CSSetShader(shader, nullptr, 0u);
        ID3D11ShaderResourceView* srvs[] = { targetSRV, logitsSRV };
        context->CSSetShaderResources(1u, 2u, srvs);
        ID3D11UnorderedAccessView* uavs[] = { outUAV };
        context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
        ID3D11Buffer* cbs[] = { outCBuffer, targetCBuffer, logitsCBuffer };
        context->CSSetConstantBuffers(0u, 3u, cbs);

        const uint32_t rows = steps * batch;
        context->Dispatch(rows, 1u, 1u);

        ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
        context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
        context->CSSetShaderResources(1u, 2u, nullSRVs);
        context->CSSetShader(nullptr, nullptr, 0u);

        if (GPUSyncCalls)
            DirectX11Manager::Instance()->WaitForGPU();
    }
};
#endif
