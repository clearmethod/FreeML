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
Datablob<T, Mat>* InitConv2DBlob(   uint32_t       _inChannels,
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

        uint32_t stride      = _blob->GetUInt("Stride");
        uint32_t dilation    = _blob->GetUInt("Dilation");
        uint32_t padding     = _blob->GetUInt("Padding");
        uint32_t outChannels = _blob->GetUInt("OutChannels");

        for(uint32_t oc = 0; oc < outChannels; oc++)
        {
            typename MatrixManager<T, Mat>::MatrixRef kernelRef = _blob->AcquireMatrix("KernelWeights_"+std::to_string(oc));
            Mat* kernelWeights = kernelRef.get();
            Conv2DSingleChannel<T>(output, input, kernelWeights, bias, oc, stride, dilation, padding);
        }
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef errorInRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* errorIn = errorInRef.get();        // Incoming gradient (dL/dY)
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();            // Input (X)
        typename MatrixManager<T, Mat>::MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOut");
        Mat* errorOut = errorOutRef.get();      // Outgoing gradient (dL/dX)
        typename MatrixManager<T, Mat>::MatrixRef bUpdateRef = _blob->AcquireMatrix("BUpdate");
        Mat* bUpdate = bUpdateRef.get();        // Bias gradient

        uint32_t stride      = _blob->GetUInt("Stride");
        uint32_t dilation    = _blob->GetUInt("Dilation");
        uint32_t padding     = _blob->GetUInt("Padding");
        uint32_t outChannels = _blob->GetUInt("OutChannels");

        if(errorOut) 
            Clear(errorOut);
        if(bUpdate)  
            Clear(bUpdate);

        for(uint32_t oc = 0; oc < outChannels; ++oc)
        {
            typename MatrixManager<T, Mat>::MatrixRef wUpdateRef = _blob->AcquireMatrix("WUpdate_" + std::to_string(oc));
            Mat* wUpdate = wUpdateRef.get();
            typename MatrixManager<T, Mat>::MatrixRef kernelRef = _blob->AcquireMatrix("KernelWeights_" + std::to_string(oc));
            Mat* kernel = kernelRef.get();
            if(wUpdate) 
                Clear(wUpdate);

            // Calculate Bias Gradient: Sum errorIn over spatial dims
            if(bUpdate)
            {
                SumSpatialDimension(bUpdate, errorIn, oc, Dims3D(oc, 0u, 0u));
            }
            Conv2DSingleChannelBackwards<T>(errorIn, input, kernel, wUpdate, errorOut, oc, stride, dilation, padding);
        }
    }
};
