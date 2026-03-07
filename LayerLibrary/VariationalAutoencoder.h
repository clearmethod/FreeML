#pragma once

#include "Layer.h"
#include "Datablob.h"
#include "Conv2D.h"

#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/MatrixBase_Functions.h"

#include <ActivationLibrary/Identity.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
 
template<class T, class Mat = MatrixCPU<T>, class ConvAct = Identity<T>>
Datablob<T, Mat>* InitVariationalAutoencoderBlob(std::vector<ConvSettings>& _convData,
                                                 uint32_t _latentChannels = 0u,
                                                 bool _initForTraining = true,
                                                 float _klWeight = 1e-4f)
{
    Datablob<T, Mat>* blob = new Datablob<T, Mat>();
    blob->Set("InitForTraining", _initForTraining ? 1u : 0u);
    blob->Set("KLWeight", _klWeight);

    const uint32_t convCount = static_cast<uint32_t>(_convData.size());
    blob->Set("LayerCount", convCount);

    for (uint32_t i = 0; i < convCount; ++i)
    {
        auto* convLayer = new Conv2D<T, ConvAct, Mat>();
        auto* convBlob = InitConv2DBlob<T, Mat>(_convData[i].inChannels,
                                                _convData[i].outChannels,
                                                _convData[i].kernelSize,
                                                1u,
                                                _convData[i].stride,
                                                _convData[i].padding,
                                                _convData[i].dilation,
                                                _initForTraining,
                                                true,
                                                _convData[i].useBias);
        blob->Set("ConvLayer_" + std::to_string(i), convLayer);
        blob->Set("ConvBlob_" + std::to_string(i), convBlob);
    }

    const uint32_t hiddenChannels = convCount > 0u ? _convData.back().outChannels : 0u;
    const uint32_t latentChannels = (_latentChannels == 0u) ? hiddenChannels : _latentChannels;
    blob->Set("LatentChannels", latentChannels);

    auto* muLayer = new Conv2D<T, Identity<T>, Mat>();
    auto* muBlob = InitConv2DBlob<T, Mat>(hiddenChannels,
                                          latentChannels,
                                          Dims3D(1u, 1u, 1u),
                                          1u,
                                          1u,
                                          0u,
                                          1u,
                                          _initForTraining,
                                          true,
                                          true);
    blob->Set("MuLayer", muLayer);
    blob->Set("MuBlob", muBlob);

    auto* logVarLayer = new Conv2D<T, Identity<T>, Mat>();
    auto* logVarBlob = InitConv2DBlob<T, Mat>(hiddenChannels,
                                              latentChannels,
                                              Dims3D(1u, 1u, 1u),
                                              1u,
                                              1u,
                                              0u,
                                              1u,
                                              _initForTraining,
                                              true,
                                              true);
    blob->Set("LogVarLayer", logVarLayer);
    blob->Set("LogVarBlob", logVarBlob);

    return blob;
}

template<class T, class ActFunc, class Mat = MatrixCPU<T>>
class VariationalAutoencoder : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

public:
    virtual std::string GetTypeName() override
    {
        return "VariationalAutoencoder";
    }

    std::vector<MatrixRef>* GetWeights(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_weightMatrices.clear();
        AppendLayerWeights(_blob, Layer<T, Mat>::m_weightMatrices);
        return &this->m_weightMatrices;
    }

    std::vector<MatrixRef>* GetGradients(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_gradientMatrices.clear();
        AppendLayerGradients(_blob, Layer<T, Mat>::m_gradientMatrices);
        return &this->m_gradientMatrices;
    }

    MatrixRef GetOutputError(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        return _blob->AcquireMatrix("ErrorOutput_0");
    }

    MatrixRef GetOutput(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        if (_index == 1u)
        {
            return _blob->AcquireMatrix("Output_1_Mu");
        }
        if (_index == 2u)
        {
            return _blob->AcquireMatrix("Output_2_LogVar");
        }
        return _blob->AcquireMatrix("Output_0_Latent");
    }

    void EnsureOutputsAllocated(Datablob<T, Mat>* _blob) override
    {
        MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        if (!input)
        {
            return;
        }

        const uint32_t convCount = _blob->GetUInt("LayerCount");
        Mat* nextInput = input;
        for (uint32_t i = 0; i < convCount; ++i)
        {
            Layer<T, Mat>* convLayer = nullptr;
            Datablob<T, Mat>* convBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "ConvLayer_" + std::to_string(i),
                                  "ConvBlob_" + std::to_string(i),
                                  convLayer,
                                  convBlob);
            if (!convLayer || !convBlob)
            {
                continue;
            }
            convLayer->SetInput(convBlob, nextInput);
            convLayer->EnsureOutputsAllocated(convBlob);
            MatrixRef convOut = convLayer->GetOutput(convBlob, 0u);
            if (convOut.get())
            {
                nextInput = convOut.get();
            }
        }

        Layer<T, Mat>* muLayer = nullptr;
        Datablob<T, Mat>* muBlob = nullptr;
        ResolveChildLayerBlob(_blob, "MuLayer", "MuBlob", muLayer, muBlob);
        Layer<T, Mat>* logVarLayer = nullptr;
        Datablob<T, Mat>* logVarBlob = nullptr;
        ResolveChildLayerBlob(_blob, "LogVarLayer", "LogVarBlob", logVarLayer, logVarBlob);
        if (!muLayer || !muBlob || !logVarLayer || !logVarBlob)
        {
            return;
        }

        muLayer->SetInput(muBlob, nextInput);
        logVarLayer->SetInput(logVarBlob, nextInput);
        muLayer->EnsureOutputsAllocated(muBlob);
        logVarLayer->EnsureOutputsAllocated(logVarBlob);

        MatrixRef muRef = muLayer->GetOutput(muBlob, 0u);
        MatrixRef logVarRef = logVarLayer->GetOutput(logVarBlob, 0u);
        if (!muRef.get() || !logVarRef.get())
        {
            return;
        }

        _blob->Set("Output_1_Mu", muRef);
        _blob->Set("Output_2_LogVar", logVarRef);

        // Reparameterization buffers: Output_0_Latent is the sampled z, not mu.
        this->EnsureMatrix(_blob, "Output_0_Latent", muRef->GetDimsX(), muRef->GetDimsY(), muRef->GetDimsZ());
        this->EnsureMatrix(_blob, "Eps",             muRef->GetDimsX(), muRef->GetDimsY(), muRef->GetDimsZ());
        // Gradient buffers for the two separate head backward passes.
        this->EnsureMatrix(_blob, "MuGrad",          muRef->GetDimsX(), muRef->GetDimsY(), muRef->GetDimsZ());
        this->EnsureMatrix(_blob, "LogVarGrad",      muRef->GetDimsX(), muRef->GetDimsY(), muRef->GetDimsZ());
        this->EnsureMatrix(_blob, "HiddenGrad",      nextInput->GetDimsX(), nextInput->GetDimsY(), nextInput->GetDimsZ());

        this->EnsureMatrix(_blob, "ErrorOutput_0", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {
        MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        if (!input)
        {
            return;
        }

        const uint32_t convCount = _blob->GetUInt("LayerCount");
        Mat* nextInput = input;
        for (uint32_t i = 0; i < convCount; ++i)
        {
            Layer<T, Mat>* convLayer = nullptr;
            Datablob<T, Mat>* convBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "ConvLayer_" + std::to_string(i),
                                  "ConvBlob_" + std::to_string(i),
                                  convLayer,
                                  convBlob);

            convLayer->SetInput(convBlob, nextInput);
            convLayer->Forward(convBlob);
            MatrixRef outRef = convLayer->GetOutput(convBlob, 0u);
            nextInput = outRef.get();
        }

        Layer<T, Mat>* muLayer = nullptr;
        Datablob<T, Mat>* muBlob = nullptr;
        ResolveChildLayerBlob(_blob, "MuLayer", "MuBlob", muLayer, muBlob);
        Layer<T, Mat>* logVarLayer = nullptr;
        Datablob<T, Mat>* logVarBlob = nullptr;
        ResolveChildLayerBlob(_blob, "LogVarLayer", "LogVarBlob", logVarLayer, logVarBlob);

        muLayer->SetInput(muBlob, nextInput);
        logVarLayer->SetInput(logVarBlob, nextInput);
        muLayer->Forward(muBlob);
        logVarLayer->Forward(logVarBlob);

        MatrixRef muRef = muLayer->GetOutput(muBlob, 0u);
        MatrixRef logVarRef = logVarLayer->GetOutput(logVarBlob, 0u);
        if (!muRef.get() || !logVarRef.get())
        {
            return;
        }

        _blob->Set("Output_1_Mu", muRef);
        _blob->Set("Output_2_LogVar", logVarRef);

        // Reparameterization trick: z = mu + exp(0.5*logvar) * eps, eps ~ N(0,I).
        // eps is stored so Backwards can use it without re-sampling.
        MatrixRef latentRef = _blob->AcquireMatrix("Output_0_Latent");
        MatrixRef epsRef    = _blob->AcquireMatrix("Eps");
        if (latentRef.get() && epsRef.get())
        {
            ReparameterizeMat(latentRef.get(), epsRef.get(), muRef.get(), logVarRef.get());
        }
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        MatrixRef errorInRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* errorIn = errorInRef.get();

        Layer<T, Mat>* muLayer = nullptr;
        Datablob<T, Mat>* muBlob = nullptr;
        ResolveChildLayerBlob(_blob, "MuLayer", "MuBlob", muLayer, muBlob);
        Layer<T, Mat>* logVarLayer = nullptr;
        Datablob<T, Mat>* logVarBlob = nullptr;
        ResolveChildLayerBlob(_blob, "LogVarLayer", "LogVarBlob", logVarLayer, logVarBlob);

        // Compute per-element gradients for both heads.
        // Convention: errorIn = -dL/dz (negative gradient, consistent with the rest of the
        // codebase where optimizer does param += lr * WUpdate and WUpdate = -dL/dW).
        //
        //   mu_grad     = -dL/dz  -  kl_weight * mu
        //   logvar_grad = -dL/dz * eps * 0.5 * sigma  -  kl_weight * 0.5 * (sigma^2 - 1)
        //
        // Note: dz[i] already equals -dL_recon/dz_i, so the KL terms are subtracted.
        // kl_weight=0 trains a plain autoencoder (best reconstruction quality for a single
        // training image); increase it when a smooth latent space is needed for sampling.
        const float kl_weight = _blob->GetFloat("KLWeight");
        MatrixRef muGradRef  = _blob->AcquireMatrix("MuGrad");
        MatrixRef lvGradRef  = _blob->AcquireMatrix("LogVarGrad");
        MatrixRef muValRef   = _blob->AcquireMatrix("Output_1_Mu");
        MatrixRef lvValRef   = _blob->AcquireMatrix("Output_2_LogVar");
        MatrixRef epsRef     = _blob->AcquireMatrix("Eps");
        if (errorIn && muValRef.get() && lvValRef.get() && epsRef.get() &&
            muGradRef.get() && lvGradRef.get())
        {
            ReparameterizeBackwardsMat(muGradRef.get(), lvGradRef.get(),
                                       errorIn,
                                       muValRef.get(), lvValRef.get(), epsRef.get(),
                                       kl_weight);
        }

        // Backward through mu head.
        if (muLayer && muBlob)
        {
            muBlob->Set("ErrorInput_0", muGradRef);
            muLayer->Backwards(muBlob);
        }

        // Backward through logvar head.
        if (logVarLayer && logVarBlob)
        {
            logVarBlob->Set("ErrorInput_0", lvGradRef);
            logVarLayer->Backwards(logVarBlob);
        }

        // Both heads share the same hidden input, so accumulate their error gradients.
        MatrixRef muHidErrRef  = muLayer     ? muLayer->GetOutputError(muBlob, 0u)         : MatrixRef{};
        MatrixRef lvHidErrRef  = logVarLayer ? logVarLayer->GetOutputError(logVarBlob, 0u) : MatrixRef{};
        MatrixRef hiddenGradRef = _blob->AcquireMatrix("HiddenGrad");

        MatrixRef nextErrRef = muHidErrRef;
        if (muHidErrRef.get() && lvHidErrRef.get() && hiddenGradRef.get())
        {
            Add(hiddenGradRef.get(), muHidErrRef.get(), lvHidErrRef.get());
            nextErrRef = hiddenGradRef;
        }
        Mat* nextError = nextErrRef.get();

        const uint32_t convCount = _blob->GetUInt("LayerCount");
        for (int32_t i = static_cast<int32_t>(convCount) - 1; i >= 0; --i)
        {
            Layer<T, Mat>* convLayer = nullptr;
            Datablob<T, Mat>* convBlob = nullptr;
            const std::string index = std::to_string(static_cast<uint32_t>(i));
            ResolveChildLayerBlob(_blob, "ConvLayer_" + index, "ConvBlob_" + index, convLayer, convBlob);

            convBlob->Set("ErrorInput_0", nextError);
            convLayer->Backwards(convBlob);
            nextErrRef = convLayer->GetOutputError(convBlob, 0u);
            nextError  = nextErrRef.get();
        }

        MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOutput_0");
        Mat* errorOut = errorOutRef.get();
        if (errorOut && nextError)
        {
            Copy(errorOut, nextError);
        }
    }

    virtual void GetSublayerPairs(std::vector<typename Layer<T, Mat>::sublayerinfo>& _out,
                                  Datablob<T, Mat>* _blob) override
    {
        const uint32_t convCount = _blob->GetUInt("LayerCount");
        for (uint32_t i = 0; i < convCount; ++i)
        {
            Layer<T, Mat>* convLayer = nullptr;
            Datablob<T, Mat>* convBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "ConvLayer_" + std::to_string(i),
                                  "ConvBlob_" + std::to_string(i),
                                  convLayer,
                                  convBlob);
            if (convLayer && convBlob)
            {
                _out.push_back(typename Layer<T, Mat>::sublayerinfo{"ConvLayer_" + std::to_string(i), convLayer, convBlob});
            }
        }

        Layer<T, Mat>* muLayer = nullptr;
        Datablob<T, Mat>* muBlob = nullptr;
        ResolveChildLayerBlob(_blob, "MuLayer", "MuBlob", muLayer, muBlob);
        if (muLayer && muBlob)
        {
            _out.push_back(typename Layer<T, Mat>::sublayerinfo{"MuLayer", muLayer, muBlob});
        }

        Layer<T, Mat>* logVarLayer = nullptr;
        Datablob<T, Mat>* logVarBlob = nullptr;
        ResolveChildLayerBlob(_blob, "LogVarLayer", "LogVarBlob", logVarLayer, logVarBlob);
        if (logVarLayer && logVarBlob)
        {
            _out.push_back(typename Layer<T, Mat>::sublayerinfo{"LogVarLayer", logVarLayer, logVarBlob});
        }
    }

private:
    static void ResolveChildLayerBlob(Datablob<T, Mat>* _blob,
                                      const std::string& _layerKey,
                                      const std::string& _blobKey,
                                      Layer<T, Mat>*& _layerOut,
                                      Datablob<T, Mat>*& _blobOut)
    {
        _layerOut = _blob ? _blob->GetLayer(_layerKey) : nullptr;
        if (!_layerOut && _blob)
        {
            _layerOut = _blob->GetLayer(_blobKey);
        }

        _blobOut = _blob ? _blob->GetBlob(_blobKey) : nullptr;
        if (!_blobOut && _blob)
        {
            _blobOut = _blob->GetBlob(_layerKey);
        }
    }

    void AppendLayerWeights(Datablob<T, Mat>* _blob, std::vector<MatrixRef>& _out)
    {
        const uint32_t convCount = _blob->GetUInt("LayerCount");
        for (uint32_t i = 0; i < convCount; ++i)
        {
            Layer<T, Mat>* convLayer = nullptr;
            Datablob<T, Mat>* convBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "ConvLayer_" + std::to_string(i),
                                  "ConvBlob_" + std::to_string(i),
                                  convLayer,
                                  convBlob);
            if (!convLayer || !convBlob)
            {
                continue;
            }
            std::vector<MatrixRef>* w = convLayer->GetWeights(convBlob);
            if (!w)
            {
                continue;
            }
            _out.insert(_out.end(), w->begin(), w->end());
        }

        Layer<T, Mat>* muLayer = nullptr;
        Datablob<T, Mat>* muBlob = nullptr;
        ResolveChildLayerBlob(_blob, "MuLayer", "MuBlob", muLayer, muBlob);
        if (muLayer && muBlob)
        {
            std::vector<MatrixRef>* w = muLayer->GetWeights(muBlob);
            if (w)
            {
                _out.insert(_out.end(), w->begin(), w->end());
            }
        }

        Layer<T, Mat>* logVarLayer = nullptr;
        Datablob<T, Mat>* logVarBlob = nullptr;
        ResolveChildLayerBlob(_blob, "LogVarLayer", "LogVarBlob", logVarLayer, logVarBlob);
        if (logVarLayer && logVarBlob)
        {
            std::vector<MatrixRef>* w = logVarLayer->GetWeights(logVarBlob);
            if (w)
            {
                _out.insert(_out.end(), w->begin(), w->end());
            }
        }
    }

    void AppendLayerGradients(Datablob<T, Mat>* _blob, std::vector<MatrixRef>& _out)
    {
        const uint32_t convCount = _blob->GetUInt("LayerCount");
        for (uint32_t i = 0; i < convCount; ++i)
        {
            Layer<T, Mat>* convLayer = nullptr;
            Datablob<T, Mat>* convBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "ConvLayer_" + std::to_string(i),
                                  "ConvBlob_" + std::to_string(i),
                                  convLayer,
                                  convBlob);
            if (!convLayer || !convBlob)
            {
                continue;
            }
            std::vector<MatrixRef>* g = convLayer->GetGradients(convBlob);
            if (!g)
            {
                continue;
            }
            _out.insert(_out.end(), g->begin(), g->end());
        }

        Layer<T, Mat>* muLayer = nullptr;
        Datablob<T, Mat>* muBlob = nullptr;
        ResolveChildLayerBlob(_blob, "MuLayer", "MuBlob", muLayer, muBlob);
        if (muLayer && muBlob)
        {
            std::vector<MatrixRef>* g = muLayer->GetGradients(muBlob);
            if (g)
            {
                _out.insert(_out.end(), g->begin(), g->end());
            }
        }

        Layer<T, Mat>* logVarLayer = nullptr;
        Datablob<T, Mat>* logVarBlob = nullptr;
        ResolveChildLayerBlob(_blob, "LogVarLayer", "LogVarBlob", logVarLayer, logVarBlob);
        if (logVarLayer && logVarBlob)
        {
            std::vector<MatrixRef>* g = logVarLayer->GetGradients(logVarBlob);
            if (g)
            {
                _out.insert(_out.end(), g->begin(), g->end());
            }
        }
    }
};
