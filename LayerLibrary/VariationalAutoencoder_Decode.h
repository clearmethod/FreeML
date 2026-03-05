#pragma once

#include "Layer.h"
#include "Datablob.h"
#include "Conv2DTranspose.h"

#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/MatrixBase_Functions.h"

#include <ActivationLibrary/Identity.h>

#include <cstdint>
#include <string>
#include <vector>

template<class T, class Mat = MatrixCPU<T>, class ConvTransposeAct = Identity<T>>
Datablob<T, Mat>* InitVariationalAutoencoderDecodeBlob(std::vector<ConvTransposeSettings>& _convTransposeData,
                                                       bool _initForTraining = true)
{
    Datablob<T, Mat>* blob = new Datablob<T, Mat>();
    blob->Set("InitForTraining", _initForTraining ? 1u : 0u);

    const uint32_t layerCount = static_cast<uint32_t>(_convTransposeData.size());
    blob->Set("LayerCount", layerCount);

    for (uint32_t i = 0; i < layerCount; ++i)
    {
        auto* convTransposeLayer = new Conv2DTranspose<T, ConvTransposeAct, Mat>();
        auto* convTransposeBlob = InitConv2DTransposeBlob<T, Mat>(_convTransposeData[i].inChannels,
                                                                  _convTransposeData[i].outChannels,
                                                                  _convTransposeData[i].kernelSize,
                                                                  1u,
                                                                  _convTransposeData[i].stride,
                                                                  _convTransposeData[i].padding,
                                                                  _convTransposeData[i].outputPadding,
                                                                  _convTransposeData[i].dilation,
                                                                  _initForTraining,
                                                                  true,
                                                                  _convTransposeData[i].useBias);
        blob->Set("ConvTransposeLayer_" + std::to_string(i), convTransposeLayer);
        blob->Set("ConvTransposeBlob_" + std::to_string(i), convTransposeBlob);
    }

    return blob;
}

template<class T, class ActFunc, class Mat = MatrixCPU<T>>
class VariationalAutoencoder_Decode : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

public:
    virtual std::string GetTypeName() override
    {
        return "VariationalAutoencoder_Decode";
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
        return _blob->AcquireMatrix("Output_0");
    }

    void EnsureOutputsAllocated(Datablob<T, Mat>* _blob) override
    {
        MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        if (!input)
        {
            return;
        }

        MatrixRef nextRef = inputRef;
        Mat* nextInput = input;
        const uint32_t layerCount = _blob->GetUInt("LayerCount");

        for (uint32_t i = 0; i < layerCount; ++i)
        {
            Layer<T, Mat>* convTransposeLayer = nullptr;
            Datablob<T, Mat>* convTransposeBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "ConvTransposeLayer_" + std::to_string(i),
                                  "ConvTransposeBlob_" + std::to_string(i),
                                  convTransposeLayer,
                                  convTransposeBlob);
            if (!convTransposeLayer || !convTransposeBlob)
            {
                continue;
            }

            convTransposeLayer->SetInput(convTransposeBlob, nextInput);
            convTransposeLayer->EnsureOutputsAllocated(convTransposeBlob);
            nextRef = convTransposeLayer->GetOutput(convTransposeBlob, 0u);
            if (nextRef.get())
            {
                nextInput = nextRef.get();
            }
        }

        if (nextRef.get())
        {
            _blob->Set("Output_0", nextRef);
        }

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

        MatrixRef nextRef = inputRef;
        Mat* nextInput = input;
        const uint32_t layerCount = _blob->GetUInt("LayerCount");

        for (uint32_t i = 0; i < layerCount; ++i)
        {
            Layer<T, Mat>* convTransposeLayer = nullptr;
            Datablob<T, Mat>* convTransposeBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "ConvTransposeLayer_" + std::to_string(i),
                                  "ConvTransposeBlob_" + std::to_string(i),
                                  convTransposeLayer,
                                  convTransposeBlob);
            if (!convTransposeLayer || !convTransposeBlob)
            {
                continue;
            }

            convTransposeLayer->SetInput(convTransposeBlob, nextInput);
            convTransposeLayer->Forward(convTransposeBlob);
            nextRef = convTransposeLayer->GetOutput(convTransposeBlob, 0u);
            if (nextRef.get())
            {
                nextInput = nextRef.get();
            }
        }

        if (nextRef.get())
        {
            _blob->Set("Output_0", nextRef);
        }
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        MatrixRef errorInRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* errorIn = errorInRef.get();
        if (!errorIn)
        {
            return;
        }

        MatrixRef nextErrRef = errorInRef;
        Mat* nextError = errorIn;

        const uint32_t layerCount = _blob->GetUInt("LayerCount");
        for (int32_t i = static_cast<int32_t>(layerCount) - 1; i >= 0; --i)
        {
            Layer<T, Mat>* convTransposeLayer = nullptr;
            Datablob<T, Mat>* convTransposeBlob = nullptr;
            const std::string index = std::to_string(static_cast<uint32_t>(i));
            ResolveChildLayerBlob(_blob,
                                  "ConvTransposeLayer_" + index,
                                  "ConvTransposeBlob_" + index,
                                  convTransposeLayer,
                                  convTransposeBlob);
            if (!convTransposeLayer || !convTransposeBlob)
            {
                continue;
            }

            convTransposeBlob->Set("ErrorInput_0", nextErrRef);
            convTransposeLayer->Backwards(convTransposeBlob);
            nextErrRef = convTransposeLayer->GetOutputError(convTransposeBlob, 0u);
            nextError = nextErrRef.get();
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
        const uint32_t layerCount = _blob->GetUInt("LayerCount");
        for (uint32_t i = 0; i < layerCount; ++i)
        {
            Layer<T, Mat>* convTransposeLayer = nullptr;
            Datablob<T, Mat>* convTransposeBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "ConvTransposeLayer_" + std::to_string(i),
                                  "ConvTransposeBlob_" + std::to_string(i),
                                  convTransposeLayer,
                                  convTransposeBlob);
            if (convTransposeLayer && convTransposeBlob)
            {
                _out.push_back(typename Layer<T, Mat>::sublayerinfo{
                    "ConvTransposeLayer_" + std::to_string(i), convTransposeLayer, convTransposeBlob});
            }
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
        const uint32_t layerCount = _blob->GetUInt("LayerCount");
        for (uint32_t i = 0; i < layerCount; ++i)
        {
            Layer<T, Mat>* convTransposeLayer = nullptr;
            Datablob<T, Mat>* convTransposeBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "ConvTransposeLayer_" + std::to_string(i),
                                  "ConvTransposeBlob_" + std::to_string(i),
                                  convTransposeLayer,
                                  convTransposeBlob);
            if (!convTransposeLayer || !convTransposeBlob)
            {
                continue;
            }

            std::vector<MatrixRef>* weights = convTransposeLayer->GetWeights(convTransposeBlob);
            if (!weights)
            {
                continue;
            }
            _out.insert(_out.end(), weights->begin(), weights->end());
        }
    }

    void AppendLayerGradients(Datablob<T, Mat>* _blob, std::vector<MatrixRef>& _out)
    {
        const uint32_t layerCount = _blob->GetUInt("LayerCount");
        for (uint32_t i = 0; i < layerCount; ++i)
        {
            Layer<T, Mat>* convTransposeLayer = nullptr;
            Datablob<T, Mat>* convTransposeBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "ConvTransposeLayer_" + std::to_string(i),
                                  "ConvTransposeBlob_" + std::to_string(i),
                                  convTransposeLayer,
                                  convTransposeBlob);
            if (!convTransposeLayer || !convTransposeBlob)
            {
                continue;
            }

            std::vector<MatrixRef>* gradients = convTransposeLayer->GetGradients(convTransposeBlob);
            if (!gradients)
            {
                continue;
            }
            _out.insert(_out.end(), gradients->begin(), gradients->end());
        }
    }
};
