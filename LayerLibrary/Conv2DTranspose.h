#pragma once

#include "Layer.h"
#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/MatrixInvalid.h"
#include "../MatrixLibrary/MatrixBase_Functions.h"
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>

#include "../ToolsLibrary/Tools.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>

#include "Datablob.h"

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitConv2DTransposeBlob(uint32_t       _inChannels,
                                          uint32_t       _outChannels,
                                          Dims3D         _kernelSize,
                                          uint32_t       _batchSize       = 1,
                                          uint32_t       _stride          = 1,
                                          uint32_t       _padding         = 0,
                                          uint32_t       _outputPadding   = 0,
                                          uint32_t       _dilation        = 1,
                                          bool           _initForTraining = true,
                                          bool           _random          = true,
                                          bool           _useBias         = true)
{
    Datablob<T, Mat>* blob = new Datablob<T, Mat>();
    MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

    for (uint32_t i = 0; i < _outChannels; ++i)
    {
        auto kernelWeights = inst.AllocateMatrix({_kernelSize.x, _kernelSize.y, _inChannels}, "kernelWeights" + std::to_string(i));
        blob->Set("KernelWeights_" + std::to_string(i), kernelWeights);

        if (_initForTraining)
        {
            auto wUpdate = inst.AllocateMatrix({_kernelSize.x, _kernelSize.y, _inChannels}, "WUpdate_" + std::to_string(i));
            blob->Set("WUpdate_" + std::to_string(i), wUpdate);
        }
    }

    if (_useBias)
    {
        auto bias = inst.AllocateMatrix({_outChannels, _batchSize}, "Bias");
        blob->Set("Bias", bias);
        if (_initForTraining)
        {
            auto bUpdate = inst.AllocateMatrix({_outChannels, _batchSize}, "BUpdate");
            blob->Set("BUpdate", bUpdate);
        }
    }

    blob->Set("KernelSizeX", _kernelSize.x);
    blob->Set("KernelSizeY", _kernelSize.y);
    blob->Set("InChannels", _inChannels);
    blob->Set("OutChannels", _outChannels);
    blob->Set("Batchsize", _batchSize);
    blob->Set("Stride", _stride);
    blob->Set("Padding", _padding);
    blob->Set("OutputPadding", _outputPadding);
    blob->Set("Dilation", _dilation);

    if (_random)
    {
        for (uint32_t i = 0; i < _outChannels; ++i)
        {
            Mat* kernel = blob->GetMatrix<Mat>("KernelWeights_" + std::to_string(i));
            if (!kernel)
            {
                continue;
            }
            const float stddev = 0.02f;
            auto normalGen = std::bind(RandomUtils::random_normal, 0.0f, stddev);
            MapFunction_Zero(kernel, normalGen);
        }
        if (_useBias)
        {
            Mat* bias = blob->GetMatrix<Mat>("Bias");
            if (bias)
            {
                Fill(bias, static_cast<T>(0));
            }
        }
    }
    else
    {
        for (uint32_t i = 0; i < _outChannels; ++i)
        {
            Mat* kernel = blob->GetMatrix<Mat>("KernelWeights_" + std::to_string(i));
            if (kernel)
            {
                Fill(kernel, static_cast<T>(0));
            }
        }
        if (_useBias)
        {
            Mat* bias = blob->GetMatrix<Mat>("Bias");
            if (bias)
            {
                Fill(bias, static_cast<T>(0));
            }
        }
    }

    return blob;
}

template<class T, class ActFunc, class Mat = MatrixCPU<T>>
class Conv2DTranspose : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

public:
    virtual std::string GetTypeName() override
    {
        return "Conv2DTranspose";
    }

    std::vector<MatrixRef>* GetWeights(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_weightMatrices.clear();
        uint32_t outChannels = _blob->GetUInt("OutChannels");
        for (uint32_t i = 0; i < outChannels; ++i)
        {
            if (MatrixRef w = _blob->AcquireMatrix("KernelWeights_" + std::to_string(i)))
            {
                Layer<T, Mat>::m_weightMatrices.push_back(w);
            }
        }
        if (MatrixRef bias = _blob->AcquireMatrix("Bias"))
        {
            Layer<T, Mat>::m_weightMatrices.push_back(bias);
        }
        return &this->m_weightMatrices;
    }

    std::vector<MatrixRef>* GetGradients(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_gradientMatrices.clear();
        uint32_t outChannels = _blob->GetUInt("OutChannels");
        for (uint32_t i = 0; i < outChannels; ++i)
        {
            if (MatrixRef w = _blob->AcquireMatrix("WUpdate_" + std::to_string(i)))
            {
                Layer<T, Mat>::m_gradientMatrices.push_back(w);
            }
        }
        if (MatrixRef b = _blob->AcquireMatrix("BUpdate"))
        {
            Layer<T, Mat>::m_gradientMatrices.push_back(b);
        }
        return &this->m_gradientMatrices;
    }

    uint32_t GetOutputErrorCount() override
    {
        return 1u;
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
        MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        if (output)
        {
            return;
        }

        MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        if (!input)
        {
            return;
        }

        if (input->GetDimsX() == 0u || input->GetDimsY() == 0u || input->GetDimsZ() == 0u)
        {
            return;
        }

        DynamicInit(_blob, input);
    }

    void DynamicInit(Datablob<T, Mat>* _blob, Mat* _input)
    {
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

        uint32_t kW = _blob->GetUInt("KernelSizeX");
        uint32_t kH = _blob->GetUInt("KernelSizeY");

        uint32_t stride = _blob->GetUInt("Stride");
        uint32_t dilation = _blob->GetUInt("Dilation");
        uint32_t padding = _blob->GetUInt("Padding");
        uint32_t outputPadding = _blob->GetUInt("OutputPadding");
        uint32_t outChannels = _blob->GetUInt("OutChannels");

        const int64_t outx64 = static_cast<int64_t>(_input->GetDimsX() - 1u) * static_cast<int64_t>(stride)
                             - static_cast<int64_t>(2u * padding)
                             + static_cast<int64_t>(dilation) * static_cast<int64_t>(kW - 1u)
                             + static_cast<int64_t>(outputPadding)
                             + 1;
        const int64_t outy64 = static_cast<int64_t>(_input->GetDimsY() - 1u) * static_cast<int64_t>(stride)
                             - static_cast<int64_t>(2u * padding)
                             + static_cast<int64_t>(dilation) * static_cast<int64_t>(kH - 1u)
                             + static_cast<int64_t>(outputPadding)
                             + 1;

        assert(outx64 > 0 && outy64 > 0);
        const uint32_t outx = static_cast<uint32_t>(outx64);
        const uint32_t outy = static_cast<uint32_t>(outy64);

        auto outputRef = inst.AllocateMatrix({outx, outy, outChannels}, "Output_0");
        _blob->Set("Output_0", outputRef);

        MatrixRef wUpdateRef = _blob->AcquireMatrix("WUpdate_0");
        if (wUpdateRef.get() != nullptr)
        {
            auto errorOut = inst.AllocateMatrix({_input->GetDimsX(), _input->GetDimsY(), _input->GetDimsZ()}, "ErrorOut");
            _blob->Set("ErrorOut", errorOut);
        }
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {
        MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        MatrixRef biasRef = _blob->AcquireMatrix("Bias");
        Mat* bias = biasRef.get();

        uint32_t stride = _blob->GetUInt("Stride");
        uint32_t dilation = _blob->GetUInt("Dilation");
        uint32_t padding = _blob->GetUInt("Padding");
        uint32_t outChannels = _blob->GetUInt("OutChannels");
        uint32_t inChannels = _blob->GetUInt("InChannels");

        assert(input);
        assert(output);
        assert(input->GetDimsZ() == inChannels && "Input channels must match Conv2DTranspose in-channels");

        Clear(output);
        for (uint32_t oc = 0; oc < outChannels; ++oc)
        {
            MatrixRef kernelRef = _blob->AcquireMatrix("KernelWeights_" + std::to_string(oc));
            Mat* kernel = kernelRef.get();
            if (!kernel)
            {
                continue;
            }
            assert(kernel->GetDimsZ() == inChannels);

            const T biasValue = bias ? bias->GetValue(oc, 0u) : static_cast<T>(0);
            for (uint32_t y = 0; y < output->GetDimsY(); ++y)
            {
                for (uint32_t x = 0; x < output->GetDimsX(); ++x)
                {
                    output->SetValue(x, y, oc, biasValue);
                }
            }

            for (uint32_t ic = 0; ic < inChannels; ++ic)
            {
                for (uint32_t inY = 0; inY < input->GetDimsY(); ++inY)
                {
                    for (uint32_t inX = 0; inX < input->GetDimsX(); ++inX)
                    {
                        const T inputValue = input->GetValue(inX, inY, ic);
                        for (uint32_t ky = 0; ky < kernel->GetDimsY(); ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel->GetDimsX(); ++kx)
                            {
                                const int32_t outX = static_cast<int32_t>(inX * stride + kx * dilation) - static_cast<int32_t>(padding);
                                const int32_t outY = static_cast<int32_t>(inY * stride + ky * dilation) - static_cast<int32_t>(padding);

                                if (outX >= 0 && outY >= 0
                                    && outX < static_cast<int32_t>(output->GetDimsX())
                                    && outY < static_cast<int32_t>(output->GetDimsY()))
                                {
                                    T val = output->GetValue(static_cast<uint32_t>(outX), static_cast<uint32_t>(outY), oc);
                                    val += inputValue * kernel->GetValue(kx, ky, ic);
                                    output->SetValue(static_cast<uint32_t>(outX), static_cast<uint32_t>(outY), oc, val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        MatrixRef errorInRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* errorIn = errorInRef.get();        // Incoming gradient (dL/dY)
        MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();            // Input (X)
        MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOut");
        Mat* errorOut = errorOutRef.get();      // Outgoing gradient (dL/dX)
        MatrixRef bUpdateRef = _blob->AcquireMatrix("BUpdate");
        Mat* bUpdate = bUpdateRef.get();        // Bias gradient

        uint32_t stride = _blob->GetUInt("Stride");
        uint32_t dilation = _blob->GetUInt("Dilation");
        uint32_t padding = _blob->GetUInt("Padding");
        uint32_t outChannels = _blob->GetUInt("OutChannels");
        uint32_t inChannels = _blob->GetUInt("InChannels");

        assert(errorIn);
        assert(input);
        assert(input->GetDimsZ() == inChannels);

        if (errorOut)
            Clear(errorOut);
        if (bUpdate)
            Clear(bUpdate);

        for (uint32_t oc = 0; oc < outChannels; ++oc)
        {
            MatrixRef wUpdateRef = _blob->AcquireMatrix("WUpdate_" + std::to_string(oc));
            Mat* wUpdate = wUpdateRef.get();
            MatrixRef kernelRef = _blob->AcquireMatrix("KernelWeights_" + std::to_string(oc));
            Mat* kernel = kernelRef.get();

            if (wUpdate)
                Clear(wUpdate);
            if (!kernel)
            {
                continue;
            }
            assert(kernel->GetDimsZ() == inChannels);

            // Calculate Bias Gradient: Sum errorIn over spatial dims.
            if (bUpdate)
            {
                SumSpatialDimension(bUpdate, errorIn, oc, Dims3D(oc, 0u, 0u));
            }

            for (uint32_t ic = 0; ic < inChannels; ++ic)
            {
                for (uint32_t inY = 0; inY < input->GetDimsY(); ++inY)
                {
                    for (uint32_t inX = 0; inX < input->GetDimsX(); ++inX)
                    {
                        const T inputValue = input->GetValue(inX, inY, ic);
                        for (uint32_t ky = 0; ky < kernel->GetDimsY(); ++ky)
                        {
                            for (uint32_t kx = 0; kx < kernel->GetDimsX(); ++kx)
                            {
                                const int32_t outX = static_cast<int32_t>(inX * stride + kx * dilation) - static_cast<int32_t>(padding);
                                const int32_t outY = static_cast<int32_t>(inY * stride + ky * dilation) - static_cast<int32_t>(padding);
                                if (outX < 0 || outY < 0
                                    || outX >= static_cast<int32_t>(errorIn->GetDimsX())
                                    || outY >= static_cast<int32_t>(errorIn->GetDimsY()))
                                {
                                    continue;
                                }

                                const T grad = errorIn->GetValue(static_cast<uint32_t>(outX), static_cast<uint32_t>(outY), oc);
                                if (wUpdate)
                                {
                                    T val = wUpdate->GetValue(kx, ky, ic);
                                    val += inputValue * grad;
                                    wUpdate->SetValue(kx, ky, ic, val);
                                }
                                if (errorOut)
                                {
                                    T val = errorOut->GetValue(inX, inY, ic);
                                    val += kernel->GetValue(kx, ky, ic) * grad;
                                    errorOut->SetValue(inX, inY, ic, val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};
