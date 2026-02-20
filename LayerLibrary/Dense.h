#pragma once

#include "Layer.h"
#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/MatrixBase_Functions.h"
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
#include <MatrixLibrary/MatrixManager.h>

#include "../ToolsLibrary/Tools.h"

#include <ActivationLibrary/Identity.h>

#include <cmath>
#include <functional>
#include <utility>
#include <type_traits>

#include <ActivationLibrary/Identity.h>

#include "Datablob.h"

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitDenseBlob(     uint32_t       _dimsX,
                                uint32_t       _dimsY = 1u,
                                uint32_t       _batchSize = 1u,
                                bool           _initForTraining = true,
                                bool           _random     = true,
                                bool           _useBias    = true,
                                bool           _transposeWeights = false)
{
    Datablob<T, Mat>* blob           = new Datablob<T, Mat>();
    MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

    const uint32_t weightX = _transposeWeights ? _dimsY : _dimsX;
    const uint32_t weightY = _transposeWeights ? _dimsX : _dimsY;
    auto weights = inst.AllocateMatrix({weightX, weightY}, "Dense_Weights");
    assert(weights.get());
    blob->Set("Dense_Weights", weights);
    blob->Set("Dense_Transposed", _transposeWeights ? 1u : 0u);
    
    if(_useBias)
    {
        auto bias = inst.AllocateMatrix({_dimsX, 1u}, "bias");
        assert(bias.get());
        blob->Set("Dense_Bias", bias);
    }

    auto output                = inst.AllocateMatrix({_dimsX , _batchSize}, "Dense_Output");
    auto outputPreActivation   = inst.AllocateMatrix({_dimsX , _batchSize}, "Dense_OutputPreActivation");
    assert(output);
    assert(outputPreActivation);

    blob->Set("Output_0", output);
    blob->Set("Dense_OutputPreActivation", outputPreActivation);

    if(_initForTraining)
    {
        auto linearOut             = inst.AllocateMatrix({_dimsX , _batchSize}, "Dense_LinearOut");
        auto delta                 = inst.AllocateMatrix({_dimsX , _batchSize}, "Dense_Delta");
        auto wUpdate               = inst.AllocateMatrix({weightX, weightY}   , "Dense_WUpdate");
        auto errorOut              = inst.AllocateMatrix({_dimsY , _batchSize}, "Dense_ErrorOut");

        if(_useBias)
        {   
            auto bUpdate               = inst.AllocateMatrix({_dimsX , 1u}, "Dense_BUpdate");
            blob->Set("Dense_BUpdate"  , bUpdate);
        }

        blob->Set("Dense_LinearOut", linearOut);
        blob->Set("Dense_Delta"    , delta);
        blob->Set("Dense_WUpdate"  , wUpdate);
        blob->Set("ErrorOut" , errorOut);
    }

    if (_random)
    {
        const float stddev = 0.02f;
        auto normalGen = std::bind(RandomUtils::random_normal, 0.0f, stddev);
        MapFunction_Zero(weights.get(), normalGen);
        if ( _useBias )
        {
            auto bias = blob->GetMatrix("Dense_Bias");
            Fill(bias, static_cast<T>(0));
        }
    }
    else
    {
        Fill(weights.get(), static_cast<T>(0));
        if ( _useBias )
        {
            auto bias = blob->GetMatrix("Dense_Bias");
            Fill(bias,static_cast<T>(0));
        }
    }

    return blob;
}

template<class T, class ActFunc, class Mat = MatrixCPU<T>, bool TransposedWeights = false>
class Dense : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

    public:

    virtual std::string GetTypeName() override
    {
        return "Dense";
    }

    static constexpr bool kIsIdentity = std::is_same_v<ActFunc, Identity<T>>;
    std::vector<MatrixRef>* GetWeights(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_weightMatrices.clear();
        if (MatrixRef weights = _blob->AcquireMatrix("Dense_Weights"))
        {
            Layer<T, Mat>::m_weightMatrices.push_back(weights);
        }
        if (MatrixRef bias = _blob->AcquireMatrix("Dense_Bias"))
        {
            Layer<T, Mat>::m_weightMatrices.push_back(bias);
        }
        return &this->m_weightMatrices;
	}

    std::vector<MatrixRef>* GetGradients(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_gradientMatrices.clear();
        if (MatrixRef wUpdate = _blob->AcquireMatrix("Dense_WUpdate"))
        {
            Layer<T, Mat>::m_gradientMatrices.push_back(wUpdate);
        }
        if (MatrixRef bUpdate = _blob->AcquireMatrix("Dense_BUpdate"))
        {
            Layer<T, Mat>::m_gradientMatrices.push_back(bUpdate);
        }
		return &this->m_gradientMatrices;
    }

    Dims3D GetWeightDims(Datablob<T, Mat>* _blob) const
    {
        if (!_blob)
        {
            return Dims3D();
        }
        Mat* weights = _blob->GetMatrix("Dense_Weights");
        if (!weights)
        {
            return Dims3D();
        }
        if constexpr (TransposedWeights)
        {
            return Dims3D(weights->GetDimsY(), weights->GetDimsX(), weights->GetDimsZ());
        }
        return weights->GetDims();
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

    virtual std::string GetMetaData()
    {
        ActFunc a;
        return a.Name();
    }

    void Forward(Datablob<T, Mat>* _blob)  override
    {
        // Get the input
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();

        // Get the variables we need from its blob.
        [[maybe_unused]] typename MatrixManager<T, Mat>::MatrixRef outputPreActivationRef = _blob->AcquireMatrix("Dense_OutputPreActivation");
        [[maybe_unused]] Mat* outputPreActivation = outputPreActivationRef.get();
        typename MatrixManager<T, Mat>::MatrixRef weightsRef = _blob->AcquireMatrix("Dense_Weights");
        Mat* weights = weightsRef.get();
        typename MatrixManager<T, Mat>::MatrixRef biasRef = _blob->AcquireMatrix("Dense_Bias");
        Mat* bias = biasRef.get();
        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();

        Mat* linearOut = kIsIdentity ? output : outputPreActivation;
        
        // Transposing the matrix to get better cache behaviour can help
        // so we have an option to initalise the weights as "transposed"
        // to handle this we need to use mul function which will read it
        // transposed in the mul.
        if constexpr (TransposedWeights)
        {
            Mul<TransposeMode::Right>(linearOut, input, weights);
        }
        else
        {
            Mul<TransposeMode::None>(linearOut, input, weights);
        }

        // Apply the bias if we are using it.
        if (bias != nullptr)
        {
            // This could be fused with above for better memory performance.
            BroadcastAdd(linearOut, linearOut, bias);
        }

        // We only run an activation function if it is not an identity function.
        if constexpr (!kIsIdentity)
        {
            ActFunc act;
            act.activateMat(output, linearOut, GetGlobalThreadPool());
        }
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef errorRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* error = errorRef.get();
        typename MatrixManager<T, Mat>::MatrixRef outputPreActivationRef = _blob->AcquireMatrix("Dense_OutputPreActivation");
        Mat* outputPreActivation = outputPreActivationRef.get();
        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef weightsRef = _blob->AcquireMatrix("Dense_Weights");
        Mat* weights = weightsRef.get();
        typename MatrixManager<T, Mat>::MatrixRef linearOutRef = _blob->AcquireMatrix("Dense_LinearOut");
        Mat* linearOut = linearOutRef.get();
        typename MatrixManager<T, Mat>::MatrixRef deltaRef = _blob->AcquireMatrix("Dense_Delta");
        Mat* delta = deltaRef.get();
        typename MatrixManager<T, Mat>::MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOut");
        Mat* errorOut = errorOutRef.get();
        typename MatrixManager<T, Mat>::MatrixRef bUpdateRef = _blob->AcquireMatrix("Dense_BUpdate");
        Mat* bUpdate = bUpdateRef.get();
        typename MatrixManager<T, Mat>::MatrixRef wUpdateRef = _blob->AcquireMatrix("Dense_WUpdate");
        Mat* wUpdate = wUpdateRef.get();
        typename MatrixManager<T, Mat>::MatrixRef lastInputRef = _blob->AcquireMatrix("Input_0");
        Mat* lastInput = lastInputRef.get();

        if constexpr (kIsIdentity)
        {
            Copy(delta, error);
        }
        else
        {
            ActFunc act;
		    act.derivativeMat(linearOut, outputPreActivation, output);
            PerElementMul(delta, error, linearOut);
        }
        if constexpr (TransposedWeights)
        {
            Mul<TransposeMode::Left, T>(wUpdate, delta, lastInput);
            if (bUpdate != nullptr)
            {
                SumColumns(bUpdate, delta);
            }
            Mul<TransposeMode::None, T>(errorOut, delta, weights);
        }
        else
        {
            Mul<TransposeMode::Left, T>(wUpdate, lastInput, delta);
            if (bUpdate != nullptr)
            {
                SumColumns(bUpdate, delta);
            }
            Mul<TransposeMode::Right, T>(errorOut, delta, weights);
        }
    }
};
