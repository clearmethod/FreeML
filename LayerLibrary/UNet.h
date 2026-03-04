#pragma once

#include "Layer.h"
#include "Datablob.h"
#include "Conv2D.h"
#include "Conv2DTranspose.h"

#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/MatrixBase_Functions.h"

#include <ActivationLibrary/Gelu.h>
#include <ActivationLibrary/Identity.h>
#include <ActivationLibrary/LeakyRelu.h>
#include <ActivationLibrary/Relu.h>
#include <ActivationLibrary/ReluOpt.h>
#include <ActivationLibrary/Sigmoid.h>
#include <ActivationLibrary/Tanh.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

inline std::string NormalizeUNetActivationName(std::string activation)
{
    std::transform(activation.begin(), activation.end(), activation.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return activation;
}

template<class T, class Mat = MatrixCPU<T>, class DefaultAct = Identity<T>>
static Layer<T, Mat>* CreateConv2DLayerForActivation(const std::string& activation)
{
    const std::string norm = NormalizeUNetActivationName(activation);
    if (norm.empty())
    {
        return new Conv2D<T, DefaultAct, Mat>();
    }
    if (norm == "gelu")
    {
        return new Conv2D<T, Gelu<T, Mat>, Mat>();
    }
    if (norm == "relu")
    {
        return new Conv2D<T, Relu<T>, Mat>();
    }
    if (norm == "relu_opt")
    {
        return new Conv2D<T, ReluOpt<T>, Mat>();
    }
    if (norm == "leaky_relu")
    {
        return new Conv2D<T, LeakyRelu<T>, Mat>();
    }
    if (norm == "sigmoid")
    {
        return new Conv2D<T, Sigmoid<T>, Mat>();
    }
    if (norm == "tanh")
    {
        return new Conv2D<T, Tanh<T>, Mat>();
    }
    return new Conv2D<T, DefaultAct, Mat>();
}

template<class T, class Mat = MatrixCPU<T>, class DefaultAct = Identity<T>>
static Layer<T, Mat>* CreateConv2DTransposeLayerForActivation(const std::string& activation)
{
    const std::string norm = NormalizeUNetActivationName(activation);
    if (norm.empty())
    {
        return new Conv2DTranspose<T, DefaultAct, Mat>();
    }
    if (norm == "gelu")
    {
        return new Conv2DTranspose<T, Gelu<T, Mat>, Mat>();
    }
    if (norm == "relu")
    {
        return new Conv2DTranspose<T, Relu<T>, Mat>();
    }
    if (norm == "relu_opt")
    {
        return new Conv2DTranspose<T, ReluOpt<T>, Mat>();
    }
    if (norm == "leaky_relu")
    {
        return new Conv2DTranspose<T, LeakyRelu<T>, Mat>();
    }
    if (norm == "sigmoid")
    {
        return new Conv2DTranspose<T, Sigmoid<T>, Mat>();
    }
    if (norm == "tanh")
    {
        return new Conv2DTranspose<T, Tanh<T>, Mat>();
    }
    return new Conv2DTranspose<T, DefaultAct, Mat>();
}

// InitUNetBlob
//
// Creates a U-Net blob with N encoder stages and N-1 decoder stages.
//
// Architecture (N=4 example):
//   Input
//     -> enc_0 (Conv2D) -> SkipOut_0 saved
//     -> enc_1 (Conv2D) -> SkipOut_1 saved
//     -> enc_2 (Conv2D) -> SkipOut_2 saved
//     -> enc_3 (Conv2D) -> bottleneck output
//     -> dec_0 (Conv2DTranspose): upsample + Add(SkipOut_2) -> dec_out_0
//     -> dec_1 (Conv2DTranspose): upsample + Add(SkipOut_1) -> dec_out_1
//     -> dec_2 (Conv2DTranspose): upsample + Add(SkipOut_0) -> Output_0
//
// Preconditions (not asserted):
//   - _decoderStages.size() == _encoderStages.size() - 1
//   - _decoderStages[j].outChannels == _encoderStages[N-2-j].outChannels
//     (decoder stage j must produce the same channel count as encoder stage N-2-j
//      so the element-wise Add is valid)

template<class T, class Mat = MatrixCPU<T>, class ConvAct = Identity<T>>
Datablob<T, Mat>* InitUNetBlob(std::vector<ConvSettings>& _encoderStages,
                                std::vector<ConvTransposeSettings>& _decoderStages,
                                bool _initForTraining = true)
{
    Datablob<T, Mat>* blob = new Datablob<T, Mat>();
    blob->Set("InitForTraining", _initForTraining ? 1u : 0u);

    const uint32_t encoderCount = static_cast<uint32_t>(_encoderStages.size());
    const uint32_t decoderCount = static_cast<uint32_t>(_decoderStages.size());
    blob->Set("EncoderCount", encoderCount);
    blob->Set("DecoderCount", decoderCount);

    for (uint32_t i = 0; i < encoderCount; ++i)
    {
        auto* encLayer = CreateConv2DLayerForActivation<T, Mat, ConvAct>(_encoderStages[i].activation);
        auto* encBlob = InitConv2DBlob<T, Mat>(_encoderStages[i].inChannels,
                                               _encoderStages[i].outChannels,
                                               _encoderStages[i].kernelSize,
                                               1u,
                                               _encoderStages[i].stride,
                                               _encoderStages[i].padding,
                                               _encoderStages[i].dilation,
                                               _initForTraining,
                                               true,
                                               _encoderStages[i].useBias);
        blob->Set("EncLayer_" + std::to_string(i), encLayer);
        blob->Set("EncBlob_" + std::to_string(i), encBlob);
    }

    for (uint32_t j = 0; j < decoderCount; ++j)
    {
        auto* decLayer = CreateConv2DTransposeLayerForActivation<T, Mat, ConvAct>(_decoderStages[j].activation);
        auto* decBlob = InitConv2DTransposeBlob<T, Mat>(_decoderStages[j].inChannels,
                                                        _decoderStages[j].outChannels,
                                                        _decoderStages[j].kernelSize,
                                                        1u,
                                                        _decoderStages[j].stride,
                                                        _decoderStages[j].padding,
                                                        _decoderStages[j].outputPadding,
                                                        _decoderStages[j].dilation,
                                                        _initForTraining,
                                                        true,
                                                        _decoderStages[j].useBias);
        blob->Set("DecLayer_" + std::to_string(j), decLayer);
        blob->Set("DecBlob_" + std::to_string(j), decBlob);
    }

    return blob;
}

template<class T, class ActFunc, class Mat = MatrixCPU<T>>
class UNet : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

public:
    virtual std::string GetTypeName() override
    {
        return "UNet";
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

        const uint32_t encoderCount = _blob->GetUInt("EncoderCount");
        const uint32_t decoderCount = _blob->GetUInt("DecoderCount");

        // Run encoder chain, recording skip output sizes
        Mat* nextInput = input;
        for (uint32_t i = 0; i < encoderCount; ++i)
        {
            Layer<T, Mat>* encLayer = nullptr;
            Datablob<T, Mat>* encBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "EncLayer_" + std::to_string(i),
                                  "EncBlob_" + std::to_string(i),
                                  encLayer,
                                  encBlob);
            if (!encLayer || !encBlob)
            {
                continue;
            }
            encLayer->SetInput(encBlob, nextInput);
            encLayer->EnsureOutputsAllocated(encBlob);
            MatrixRef outRef = encLayer->GetOutput(encBlob, 0u);
            if (outRef.get())
            {
                nextInput = outRef.get();
            }

            // Allocate SkipError buffer for encoder stages that have a skip connection (0..N-2)
            if (i < encoderCount - 1u && outRef.get())
            {
                this->EnsureMatrix(_blob,
                                   ("SkipError_" + std::to_string(i)).c_str(),
                                   outRef->GetDimsX(),
                                   outRef->GetDimsY(),
                                   outRef->GetDimsZ());
            }
        }

        // Run decoder chain
        for (uint32_t j = 0; j < decoderCount; ++j)
        {
            Layer<T, Mat>* decLayer = nullptr;
            Datablob<T, Mat>* decBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "DecLayer_" + std::to_string(j),
                                  "DecBlob_" + std::to_string(j),
                                  decLayer,
                                  decBlob);
            if (!decLayer || !decBlob)
            {
                continue;
            }
            decLayer->SetInput(decBlob, nextInput);
            decLayer->EnsureOutputsAllocated(decBlob);
            MatrixRef outRef = decLayer->GetOutput(decBlob, 0u);
            if (outRef.get())
            {
                nextInput = outRef.get();
            }
        }

        // Output_0 = last decoder output
        if (nextInput && nextInput != input)
        {
            // nextInput is pointing to the last decoder's output matrix; wrap as a ref
            // We need the actual MatrixRef — re-fetch from the last decoder
            Layer<T, Mat>* lastDecLayer = nullptr;
            Datablob<T, Mat>* lastDecBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "DecLayer_" + std::to_string(decoderCount - 1u),
                                  "DecBlob_" + std::to_string(decoderCount - 1u),
                                  lastDecLayer,
                                  lastDecBlob);
            if (lastDecLayer && lastDecBlob)
            {
                MatrixRef lastOut = lastDecLayer->GetOutput(lastDecBlob, 0u);
                if (lastOut.get())
                {
                    _blob->Set("Output_0", lastOut);
                }
            }
        }

        // ErrorOut: error propagated to the input
        this->EnsureMatrix(_blob, "ErrorOut",
                           input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {
        MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        if (!input)
        {
            return;
        }

        const uint32_t encoderCount = _blob->GetUInt("EncoderCount");
        const uint32_t decoderCount = _blob->GetUInt("DecoderCount");

        // Encoder forward — save skips for stages 0..N-2
        Mat* nextInput = input;
        for (uint32_t i = 0; i < encoderCount; ++i)
        {
            Layer<T, Mat>* encLayer = nullptr;
            Datablob<T, Mat>* encBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "EncLayer_" + std::to_string(i),
                                  "EncBlob_" + std::to_string(i),
                                  encLayer,
                                  encBlob);
            if (!encLayer || !encBlob)
            {
                LOG_ERROR() << "FAILED TO GET LAYER OR BLOB in encoder. UNet::Forward";
                continue;
            }
            encLayer->SetInput(encBlob, nextInput);
            encLayer->Forward(encBlob);
            MatrixRef outRef = encLayer->GetOutput(encBlob, 0u);
            if (outRef.get())
            {
                nextInput = outRef.get();
            }

            // Save skip for all encoder stages except the deepest (bottleneck)
            if (i < encoderCount - 1u)
            {
                _blob->Set("SkipOutput_" + std::to_string(i), outRef);
            }
        }

        // Decoder forward — dec_j adds SkipOutput_{N-2-j}
        for (uint32_t j = 0; j < decoderCount; ++j)
        {
            Layer<T, Mat>* decLayer = nullptr;
            Datablob<T, Mat>* decBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "DecLayer_" + std::to_string(j),
                                  "DecBlob_" + std::to_string(j),
                                  decLayer,
                                  decBlob);
            if (!decLayer || !decBlob)
            {
                LOG_ERROR() << "FAILED TO GET LAYER OR BLOB in decoder. UNet::Forward";
                continue;
            }
            decLayer->SetInput(decBlob, nextInput);
            decLayer->Forward(decBlob);
            MatrixRef outRef = decLayer->GetOutput(decBlob, 0u);
            if (!outRef.get())
            {
                LOG_ERROR() << "FAILED TO GET decoder output. UNet::Forward";
                continue;
            }

            // Add skip connection: dec_j merges with enc_{N-2-j}
            const uint32_t skipIndex = (encoderCount - 2u) - j;
            MatrixRef skipRef = _blob->AcquireMatrix("SkipOutput_" + std::to_string(skipIndex));
            if (skipRef.get())
            {
                Add(outRef.get(), outRef.get(), skipRef.get());
            }

            nextInput = outRef.get();
        }

        // Capture Output_0 from the last decoder
        if (decoderCount > 0u)
        {
            Layer<T, Mat>* lastDecLayer = nullptr;
            Datablob<T, Mat>* lastDecBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "DecLayer_" + std::to_string(decoderCount - 1u),
                                  "DecBlob_" + std::to_string(decoderCount - 1u),
                                  lastDecLayer,
                                  lastDecBlob);
            if (lastDecLayer && lastDecBlob)
            {
                MatrixRef lastOut = lastDecLayer->GetOutput(lastDecBlob, 0u);
                if (lastOut.get())
                {
                    _blob->Set("Output_0", lastOut);
                }
            }
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

        const uint32_t encoderCount = _blob->GetUInt("EncoderCount");
        const uint32_t decoderCount = _blob->GetUInt("DecoderCount");

        // ---- Decoder backward (shallowest to deepest: j = N-2 down to 0) ----
        // For each dec_j:
        //   - The addition skip error = the incoming error (d(a+b)/db = 1)
        //   - Store it in SkipError_{N-2-j}
        //   - Run ConvTranspose backward to get error for the previous stage
        MatrixRef currentErrRef = errorInRef;
        for (int32_t j = static_cast<int32_t>(decoderCount) - 1; j >= 0; --j)
        {
            const uint32_t jIdx = static_cast<uint32_t>(j);
            Layer<T, Mat>* decLayer = nullptr;
            Datablob<T, Mat>* decBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "DecLayer_" + std::to_string(jIdx),
                                  "DecBlob_"  + std::to_string(jIdx),
                                  decLayer,
                                  decBlob);
            if (!decLayer || !decBlob)
            {
                LOG_ERROR() << "FAILED TO GET LAYER OR BLOB in encoder. UNet::Backwards";
                continue;
            }

            // Skip error for enc_{N-2-j}: same as incoming error to this decoder stage
            const uint32_t skipIndex = (encoderCount - 2u) - jIdx;
            MatrixRef skipErrRef = _blob->AcquireMatrix("SkipError_" + std::to_string(skipIndex));
            if (skipErrRef.get() && currentErrRef.get())
            {
                Copy(skipErrRef.get(), currentErrRef.get());
            }

            decBlob->Set("ErrorInput_0", currentErrRef);
            decLayer->Backwards(decBlob);
            currentErrRef = decLayer->GetOutputError(decBlob, 0u);
        }

        // currentErrRef is now the error for enc_{N-1} (bottleneck) output

        // ---- Encoder backward (deepest to shallowest: i = N-1 down to 0) ----
        // For enc_{N-1} (bottleneck): no skip error, use currentErrRef directly.
        // For enc_i (i = 0..N-2): accumulate skip error in-place into SkipError_i
        //   (SkipError_i += currentErr), then use SkipError_i as the error input.
        //   SkipError_i is already allocated at the correct dims (enc_i, output dims).
        for (int32_t i = static_cast<int32_t>(encoderCount) - 1; i >= 0; --i)
        {
            const uint32_t iIdx = static_cast<uint32_t>(i);
            Layer<T, Mat>* encLayer = nullptr;
            Datablob<T, Mat>* encBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "EncLayer_" + std::to_string(iIdx),
                                  "EncBlob_" + std::to_string(iIdx),
                                  encLayer,
                                  encBlob);
            if (!encLayer || !encBlob)
            {
                LOG_ERROR() << "FAILED TO GET LAYER OR BLOB in decoder. UNet::Backwards";
                continue;
            }

            MatrixRef errorForStage = currentErrRef;

            // Non-bottleneck encoder stages: add chain error into the skip error buffer
            if (iIdx < encoderCount - 1u)
            {
                MatrixRef skipErrRef = _blob->AcquireMatrix("SkipError_" + std::to_string(iIdx));
                if (skipErrRef.get() && currentErrRef.get())
                {
                    Add(skipErrRef.get(), skipErrRef.get(), currentErrRef.get() );
                    errorForStage = skipErrRef;
                }
            }

            encBlob->Set("ErrorInput_0", errorForStage);
            encLayer->Backwards(encBlob);
            currentErrRef = encLayer->GetOutputError(encBlob, 0u);
        }

        // Copy final error to ErrorOut
        MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOut");
        if (errorOutRef.get() && currentErrRef.get())
        {
            Copy(errorOutRef.get(), currentErrRef.get());
        }
    }

    virtual void GetSublayerPairs(std::vector<typename Layer<T, Mat>::sublayerinfo>& _out,
                                  Datablob<T, Mat>* _blob) override
    {
        const uint32_t encoderCount = _blob->GetUInt("EncoderCount");
        for (uint32_t i = 0; i < encoderCount; ++i)
        {
            Layer<T, Mat>* encLayer = nullptr;
            Datablob<T, Mat>* encBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "EncLayer_" + std::to_string(i),
                                  "EncBlob_" + std::to_string(i),
                                  encLayer,
                                  encBlob);
            if (encLayer && encBlob)
            {
                _out.push_back(typename Layer<T, Mat>::sublayerinfo{
                    "EncLayer_" + std::to_string(i), encLayer, encBlob});
            }
        }

        const uint32_t decoderCount = _blob->GetUInt("DecoderCount");
        for (uint32_t j = 0; j < decoderCount; ++j)
        {
            Layer<T, Mat>* decLayer = nullptr;
            Datablob<T, Mat>* decBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "DecLayer_" + std::to_string(j),
                                  "DecBlob_" + std::to_string(j),
                                  decLayer,
                                  decBlob);
            if (decLayer && decBlob)
            {
                _out.push_back(typename Layer<T, Mat>::sublayerinfo{
                    "DecLayer_" + std::to_string(j), decLayer, decBlob});
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
        const uint32_t encoderCount = _blob->GetUInt("EncoderCount");
        for (uint32_t i = 0; i < encoderCount; ++i)
        {
            Layer<T, Mat>* encLayer = nullptr;
            Datablob<T, Mat>* encBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "EncLayer_" + std::to_string(i),
                                  "EncBlob_" + std::to_string(i),
                                  encLayer,
                                  encBlob);
            if (!encLayer || !encBlob)
            {
                continue;
            }
            std::vector<MatrixRef>* w = encLayer->GetWeights(encBlob);
            if (w)
            {
                _out.insert(_out.end(), w->begin(), w->end());
            }
        }

        const uint32_t decoderCount = _blob->GetUInt("DecoderCount");
        for (uint32_t j = 0; j < decoderCount; ++j)
        {
            Layer<T, Mat>* decLayer = nullptr;
            Datablob<T, Mat>* decBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "DecLayer_" + std::to_string(j),
                                  "DecBlob_" + std::to_string(j),
                                  decLayer,
                                  decBlob);
            if (!decLayer || !decBlob)
            {
                continue;
            }
            std::vector<MatrixRef>* w = decLayer->GetWeights(decBlob);
            if (w)
            {
                _out.insert(_out.end(), w->begin(), w->end());
            }
        }
    }

    void AppendLayerGradients(Datablob<T, Mat>* _blob, std::vector<MatrixRef>& _out)
    {
        const uint32_t encoderCount = _blob->GetUInt("EncoderCount");
        for (uint32_t i = 0; i < encoderCount; ++i)
        {
            Layer<T, Mat>* encLayer = nullptr;
            Datablob<T, Mat>* encBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "EncLayer_" + std::to_string(i),
                                  "EncBlob_" + std::to_string(i),
                                  encLayer,
                                  encBlob);
            if (!encLayer || !encBlob)
            {
                continue;
            }
            std::vector<MatrixRef>* g = encLayer->GetGradients(encBlob);
            if (g)
            {
                _out.insert(_out.end(), g->begin(), g->end());
            }
        }

        const uint32_t decoderCount = _blob->GetUInt("DecoderCount");
        for (uint32_t j = 0; j < decoderCount; ++j)
        {
            Layer<T, Mat>* decLayer = nullptr;
            Datablob<T, Mat>* decBlob = nullptr;
            ResolveChildLayerBlob(_blob,
                                  "DecLayer_" + std::to_string(j),
                                  "DecBlob_" + std::to_string(j),
                                  decLayer,
                                  decBlob);
            if (!decLayer || !decBlob)
            {
                continue;
            }
            std::vector<MatrixRef>* g = decLayer->GetGradients(decBlob);
            if (g)
            {
                _out.insert(_out.end(), g->begin(), g->end());
            }
        }
    }
};
