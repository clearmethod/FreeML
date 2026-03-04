#pragma once

#include "Layer.h"
#include "Datablob.h"
#include "Conv2D.h"

#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/MatrixBase_Functions.h"

#include <ActivationLibrary/Identity.h>

#include <cstdint>
#include <string>
#include <vector>
 
struct ConvSettings
{
    uint32_t inChannels = 0u;
    uint32_t outChannels = 0u;
    Dims3D   kernelSize = Dims3D(1u, 1u, 1u);
    uint32_t stride = 1u;
    uint32_t padding = 0u;
    uint32_t dilation = 1u;
    bool     useBias = true;
};

template<class T, class Mat = MatrixCPU<T>, class ConvAct = Identity<T>>
Datablob<T, Mat>* InitVariationalAutoencoderBlob(std::vector<ConvSettings>& _convData,
                                                 uint32_t _latentChannels = 0u,
                                                 bool _initForTraining = true)
{
    Datablob<T, Mat>* blob = new Datablob<T, Mat>();
    blob->Set("InitForTraining", _initForTraining ? 1u : 0u);

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
        return _blob->AcquireMatrix("ErrorOut");
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
            Layer<T, Mat>* convLayer = _blob->GetLayer("ConvLayer_" + std::to_string(i));
            Datablob<T, Mat>* convBlob = _blob->GetBlob("ConvBlob_" + std::to_string(i));
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

        Layer<T, Mat>* muLayer = _blob->GetLayer("MuLayer");
        Datablob<T, Mat>* muBlob = _blob->GetBlob("MuBlob");
        Layer<T, Mat>* logVarLayer = _blob->GetLayer("LogVarLayer");
        Datablob<T, Mat>* logVarBlob = _blob->GetBlob("LogVarBlob");
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
        _blob->Set("Output_0_Latent", muRef);

        this->EnsureMatrix(_blob, "ErrorOut", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
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
            Layer<T, Mat>* convLayer = _blob->GetLayer("ConvLayer_" + std::to_string(i));
            Datablob<T, Mat>* convBlob = _blob->GetBlob("ConvBlob_" + std::to_string(i));

            convLayer->SetInput(convBlob, nextInput);
            convLayer->Forward(convBlob);
            MatrixRef outRef = convLayer->GetOutput(convBlob, 0u);
            nextInput = outRef.get();
        }

        Layer<T, Mat>* muLayer       = _blob->GetLayer("MuLayer");
        Datablob<T, Mat>* muBlob     = _blob->GetBlob("MuBlob");
        Layer<T, Mat>* logVarLayer   = _blob->GetLayer("LogVarLayer");
        Datablob<T, Mat>* logVarBlob = _blob->GetBlob("LogVarBlob");

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
        _blob->Set("Output_0_Latent", muRef);
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        MatrixRef errorInRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* errorIn = errorInRef.get();
        Layer<T, Mat>*    muLayer = _blob->GetLayer("MuLayer");
        Datablob<T, Mat>* muBlob  = _blob->GetBlob("MuBlob");

        muBlob->Set("ErrorInput_0", errorInRef);
        muLayer->Backwards(muBlob);

        MatrixRef nextErrRef = muLayer->GetOutputError(muBlob, 0u);
        Mat* nextError       = nextErrRef.get();

        const uint32_t convCount = _blob->GetUInt("LayerCount");
        for (int32_t i = static_cast<int32_t>(convCount) - 1; i >= 0; --i)
        {
            Layer<T, Mat>* convLayer = _blob->GetLayer("ConvLayer_" + std::to_string(static_cast<uint32_t>(i)));
            Datablob<T, Mat>* convBlob = _blob->GetBlob("ConvBlob_" + std::to_string(static_cast<uint32_t>(i)));

            convBlob->Set("ErrorInput_0", nextError);
            convLayer->Backwards(convBlob);
            nextErrRef = convLayer->GetOutputError(convBlob, 0u);
            nextError  = nextErrRef.get();
        }

        MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOut");
        Mat* errorOut = errorOutRef.get();
        Copy(errorOut, nextError);
    }

private:
    void AppendLayerWeights(Datablob<T, Mat>* _blob, std::vector<MatrixRef>& _out)
    {
        const uint32_t convCount = _blob->GetUInt("LayerCount");
        for (uint32_t i = 0; i < convCount; ++i)
        {
            Layer<T, Mat>* convLayer = _blob->GetLayer("ConvLayer_" + std::to_string(i));
            Datablob<T, Mat>* convBlob = _blob->GetBlob("ConvBlob_" + std::to_string(i));
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

        Layer<T, Mat>* muLayer = _blob->GetLayer("MuLayer");
        Datablob<T, Mat>* muBlob = _blob->GetBlob("MuBlob");
        if (muLayer && muBlob)
        {
            std::vector<MatrixRef>* w = muLayer->GetWeights(muBlob);
            if (w)
            {
                _out.insert(_out.end(), w->begin(), w->end());
            }
        }

        Layer<T, Mat>* logVarLayer = _blob->GetLayer("LogVarLayer");
        Datablob<T, Mat>* logVarBlob = _blob->GetBlob("LogVarBlob");
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
            Layer<T, Mat>* convLayer = _blob->GetLayer("ConvLayer_" + std::to_string(i));
            Datablob<T, Mat>* convBlob = _blob->GetBlob("ConvBlob_" + std::to_string(i));
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

        Layer<T, Mat>* muLayer = _blob->GetLayer("MuLayer");
        Datablob<T, Mat>* muBlob = _blob->GetBlob("MuBlob");
        if (muLayer && muBlob)
        {
            std::vector<MatrixRef>* g = muLayer->GetGradients(muBlob);
            if (g)
            {
                _out.insert(_out.end(), g->begin(), g->end());
            }
        }

        Layer<T, Mat>* logVarLayer = _blob->GetLayer("LogVarLayer");
        Datablob<T, Mat>* logVarBlob = _blob->GetBlob("LogVarBlob");
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
