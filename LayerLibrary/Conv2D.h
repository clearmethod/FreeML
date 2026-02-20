#pragma once

#include "Layer.h"
#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/MatrixInvalid.h"
#include "../MatrixLibrary/MatrixBase_Functions.h"
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>


#include "../ToolsLibrary/Tools.h"

#include <cmath>
#include <functional>
#include <utility>

#include "Datablob.h"

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitConv2DBlob(  uint32_t       _inChannels,
                                uint32_t       _outChannels,
                                Dims3D         _kernelSize,
                                uint32_t       _batchSize       = 1,
                                uint32_t       _stride          = 1,
                                uint32_t       _padding         = 0,
                                uint32_t       _dilation        = 1,
                                bool           _initForTraining = true,
                                bool           _random          = true,
                                bool           _useBias         = true)
{
    Datablob<T, Mat>* blob = new Datablob<T, Mat>();
    MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

    for(uint32_t i = 0; i < _outChannels; ++i)
    {
        auto kernelWeights = inst.AllocateMatrix({_kernelSize.x, _kernelSize.y, _inChannels}, "kernelWeights" + std::to_string(i));
        blob->Set("KernelWeights_"+std::to_string(i), kernelWeights);

        if(_initForTraining)
        {
            auto wUpdate = inst.AllocateMatrix({_kernelSize.x, _kernelSize.y, _inChannels}, "WUpdate_" + std::to_string(i));
            blob->Set("WUpdate_"+std::to_string(i), wUpdate);
        }
    }

    if(_useBias)
    {
        auto bias = inst.AllocateMatrix({_outChannels, _batchSize}, "Bias");
        blob->Set("Bias", bias);
        if(_initForTraining)
        {
            auto bUpdate = inst.AllocateMatrix({_outChannels, _batchSize}, "BUpdate");
            blob->Set("BUpdate", bUpdate);
        }
    }

    blob->Set("KernelSizeX", _kernelSize.x);
    blob->Set("KernelSizeY", _kernelSize.y);
    blob->Set("InChannels" , _inChannels  );
    blob->Set("OutChannels", _outChannels );
    blob->Set("Batchsize"  , _batchSize   );
    blob->Set("Stride"     , _stride      );
    blob->Set("Padding"    , _padding     );
    blob->Set("Dilation"   , _dilation    );

    return blob;
}

template<class T, class ActFunc, class Mat = MatrixCPU<T>>
class Conv2D : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

    public:
    virtual std::string GetTypeName() override
    {
        return "Conv2D";
    }

    std::vector<MatrixRef>* GetWeights(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_weightMatrices.clear();
        uint32_t outChannels = _blob->GetUInt("OutChannels");
        for(uint32_t i = 0; i < outChannels; ++i)
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
        for(uint32_t i = 0; i < outChannels; ++i)
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
        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        if (output)
        {
            return;
        }

        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
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
        // Allocate an output matrix.
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

        // Get dims from first kernel or blob metadata
        uint32_t kW = _blob->GetUInt("KernelSizeX");
        uint32_t kH = _blob->GetUInt("KernelSizeY");

        assert(_input);

        uint32_t stride      = _blob->GetUInt("Stride");
        uint32_t dilation    = _blob->GetUInt("Dilation");
        uint32_t padding     = _blob->GetUInt("Padding");
        uint32_t outChannels = _blob->GetUInt("OutChannels");

        uint32_t outx   = floor((_input->GetDimsX() + 2u*padding - dilation*(kW-1u) - 1u) / stride) + 1u;
        uint32_t outy   = floor((_input->GetDimsY() + 2u*padding - dilation*(kH-1u) - 1u) / stride) + 1u;

        auto outputRef = inst.AllocateMatrix({outx, outy, outChannels}, "Output_0");
        _blob->Set("Output_0", outputRef);

        // If we have WUpdate_0, we assume we are training and need to backpropagate error to input
        typename MatrixManager<T, Mat>::MatrixRef wUpdateRef = _blob->AcquireMatrix("WUpdate_0");
        if (wUpdateRef.get() != nullptr)
        {
            auto errorOut = inst.AllocateMatrix({_input->GetDimsX(), _input->GetDimsY(), _input->GetDimsZ()}, "ErrorOut");
            _blob->Set("ErrorOut", errorOut);
        }
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef biasRef = _blob->AcquireMatrix("Bias");
        Mat* bias = biasRef.get();
        assert(output);
        assert(input);

        uint32_t stride   = _blob->GetUInt("Stride");
        uint32_t dilation = _blob->GetUInt("Dilation");
        uint32_t padding  = _blob->GetUInt("Padding");
        uint32_t outChannels = _blob->GetUInt("OutChannels");
        assert(input->GetDimsZ() == _blob->GetUInt("InChannels") && "Input channels must match Kernel channels");

        for(uint32_t oc = 0; oc < outChannels; oc++)
        {
            typename MatrixManager<T, Mat>::MatrixRef kernelRef = _blob->AcquireMatrix("KernelWeights_"+std::to_string(oc));
            Mat* kernelWeights = kernelRef.get();

            // Loop Kernel over the input and write to output.
            for(uint32_t y = 0; y < output->GetDimsY(); y++)
            {
                for(uint32_t x = 0; x < output->GetDimsX(); x++)
                {
                    // Loop over kernel
                    T sum = T(0);
                    for(uint32_t kz = 0; kz < kernelWeights->GetDimsZ(); kz++)
                    {
                        for(uint32_t ky = 0; ky < kernelWeights->GetDimsY(); ky++)
                        {
                            for(uint32_t kx = 0; kx < kernelWeights->GetDimsX(); kx++)
                            {
                                // Handle Stride and dilation.
                                const int32_t inX = static_cast<int32_t>(x * stride + kx * dilation)
                                                    - static_cast<int32_t>(padding);
                                const int32_t inY = static_cast<int32_t>(y * stride + ky * dilation)
                                                    - static_cast<int32_t>(padding);
                                // Handing padding. If we are outsize of the range - assuming 0 added.
                                if (inX >= 0 && inY >= 0
                                    && inX < static_cast<int32_t>(input->GetDimsX())
                                    && inY < static_cast<int32_t>(input->GetDimsY()))
                                {
                                    sum += kernelWeights->GetValue(kx, ky, kz)
                                        * input->GetValue(static_cast<uint32_t>(inX),
                                                            static_cast<uint32_t>(inY),
                                                            kz);
                                }
                            }
                        }
                    }

                    if(bias) sum += bias->GetValue(oc, 0);
                    output->SetValue(x,y,oc, sum);
                }
            }
        }
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef errorInRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* errorIn = errorInRef.get(); // Incoming gradient (dL/dY)
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();      // Input (X)
        typename MatrixManager<T, Mat>::MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOut");
        Mat* errorOut = errorOutRef.get();    // Outgoing gradient (dL/dX)
        typename MatrixManager<T, Mat>::MatrixRef bUpdateRef = _blob->AcquireMatrix("BUpdate");
        Mat* bUpdate = bUpdateRef.get();     // Bias gradient

        assert(errorIn);
        assert(input);

        uint32_t stride      = _blob->GetUInt("Stride");
        uint32_t dilation    = _blob->GetUInt("Dilation");
        uint32_t padding     = _blob->GetUInt("Padding");
        uint32_t outChannels = _blob->GetUInt("OutChannels");
        uint32_t inChannels  = _blob->GetUInt("InChannels");

        if(errorOut) Fill(errorOut, static_cast<T>(0));
        if(bUpdate)  Fill(bUpdate, static_cast<T>(0));

        for(uint32_t oc = 0; oc < outChannels; ++oc)
        {
            typename MatrixManager<T, Mat>::MatrixRef wUpdateRef = _blob->AcquireMatrix("WUpdate_" + std::to_string(oc));
            Mat* wUpdate = wUpdateRef.get();
            typename MatrixManager<T, Mat>::MatrixRef kernelRef = _blob->AcquireMatrix("KernelWeights_" + std::to_string(oc));
            Mat* kernel = kernelRef.get();
            if(wUpdate) Fill(wUpdate, static_cast<T>(0));

            // Calculate Bias Gradient: Sum errorIn over spatial dims
            if(bUpdate)
            {
                T sum = static_cast<T>(0);
                for(uint32_t y = 0; y < errorIn->GetDimsY(); ++y)
                {
                    for(uint32_t x = 0; x < errorIn->GetDimsX(); ++x)
                    {
                        sum += errorIn->GetValue(x, y, oc);
                    }
                }
                bUpdate->SetValue(oc, 0, sum);
            }

            // Calculate Weight and Input Gradients
            for(uint32_t y = 0; y < errorIn->GetDimsY(); ++y)
            {
                for(uint32_t x = 0; x < errorIn->GetDimsX(); ++x)
                {
                    T grad = errorIn->GetValue(x, y, oc);
                    
                    for(uint32_t kz = 0; kz < inChannels; ++kz)
                    {
                        for(uint32_t ky = 0; ky < kernel->GetDimsY(); ++ky)
                        {
                            for(uint32_t kx = 0; kx < kernel->GetDimsX(); ++kx)
                            {
                                int32_t inX = x * stride + kx * dilation - padding;
                                int32_t inY = y * stride + ky * dilation - padding;

                                if(inX >= 0 && inY >= 0 && inX < (int32_t)input->GetDimsX() && inY < (int32_t)input->GetDimsY())
                                {
                                    if(wUpdate)
                                    {
                                        T val = wUpdate->GetValue(kx, ky, kz);
                                        val += input->GetValue(inX, inY, kz) * grad;
                                        wUpdate->SetValue(kx, ky, kz, val);
                                    }
                                    if(errorOut)
                                    {
                                        T val = errorOut->GetValue(inX, inY, kz);
                                        val += kernel->GetValue(kx, ky, kz) * grad;
                                        errorOut->SetValue(inX, inY, kz, val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};
