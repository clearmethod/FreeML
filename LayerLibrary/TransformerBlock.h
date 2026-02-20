#pragma once

#include "Layer.h"

#include <MatrixLibrary/MatrixManager.h>
#include <MatrixLibrary/MatrixBase.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
#include <MatrixLibrary/CPU/MatrixCPU.h>

#include <LayerLibrary/CausalSelfAttentionLayer.h>
#include <LayerLibrary/Dense.h>
#include <LayerLibrary/DropoutLayer.h>
#include <LayerLibrary/LayerNorm.h>

#include <ActivationLibrary/Gelu.h>
#include <ActivationLibrary/Identity.h>

#include <cmath>
#include <functional>
#include <utility>

#include "Datablob.h"

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitTransformerBlockBlob(  uint32_t _n_embed,
                                             uint32_t _n_head,
                                             float    _dropout         = 0.0,
                                             uint32_t _block_size      = 1024,
                                             uint32_t _batch_size      = 1,
                                             bool     _initForTraining = true,
                                             bool     _useBias         = true,
                                             float    _residProjScale  = 1.0f )
{
    Datablob<T, Mat>* blob           = new Datablob<T, Mat>();
    blob->Set("TrainingEnabled"     , _initForTraining  ? 1u : 0u);
    blob->Set("UseBias"             , _useBias          ? 1u : 0u);

    blob->Set("Init_NEmbed"         , _n_embed);
    blob->Set("Init_NHead"          , _n_head);
    blob->Set("Init_Dropout"        , _dropout);
    blob->Set("Init_BlockSize"      , _block_size);
    blob->Set("Init_BatchSize"      , _batch_size);
    blob->Set("Init_ResidProjScale" , _residProjScale);

    return blob;
}

template<class T, class Mat = MatrixCPU<T>>
class TransformerBlockLayer : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

    public:

    TransformerBlockLayer()
    {
        m_layerNorm0    = new LayerNorm<T,Mat>();
        m_layerNorm1    = new LayerNorm<T,Mat>();
        m_attention     = new CausalSelfAttentionLayer<T, Identity<T>, Mat>();
        m_denseFC       = new Dense<T,Gelu<T,Mat>, Mat>();
        m_denseProj     = new Dense<T,Identity<T>, Mat>();
        m_mlpDropout    = new DropoutLayer<T, Mat>();
    }

    ~TransformerBlockLayer()
    {
        delete m_layerNorm0;
        delete m_layerNorm1;
        delete m_attention;
        delete m_denseFC;
        delete m_denseProj;
        delete m_mlpDropout;
    }

    virtual std::string GetTypeName() override
    {
        return "TransformerBlock";
    }

    std::vector<MatrixRef>* GetWeights(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_weightMatrices.clear();
        if (Datablob<T, Mat>* ln0 = _blob->GetBlob("LayerNorm0Data"))
        {
            auto* weights = m_layerNorm0->GetWeights(ln0);
            Layer<T, Mat>::m_weightMatrices.insert(Layer<T, Mat>::m_weightMatrices.end(),
                                                   weights->begin(),
                                                   weights->end());
        }
        if (Datablob<T, Mat>* attention = _blob->GetBlob("AttentionLayerData"))
        {
            auto* weights = m_attention->GetWeights(attention);
            Layer<T, Mat>::m_weightMatrices.insert(Layer<T, Mat>::m_weightMatrices.end(),
                                                   weights->begin(),
                                                   weights->end());
        }
        if (Datablob<T, Mat>* ln1 = _blob->GetBlob("LayerNorm1Data"))
        {
            auto* weights = m_layerNorm1->GetWeights(ln1);
            Layer<T, Mat>::m_weightMatrices.insert(Layer<T, Mat>::m_weightMatrices.end(),
                                                   weights->begin(),
                                                   weights->end());
        }
        if (Datablob<T, Mat>* denseFC = _blob->GetBlob("DenseFCData"))
        {
            auto* weights = m_denseFC->GetWeights(denseFC);
            Layer<T, Mat>::m_weightMatrices.insert(Layer<T, Mat>::m_weightMatrices.end(),
                                                   weights->begin(),
                                                   weights->end());
        }
        if (Datablob<T, Mat>* denseProj = _blob->GetBlob("DenseProjData"))
        {
            auto* weights = m_denseProj->GetWeights(denseProj);
            Layer<T, Mat>::m_weightMatrices.insert(Layer<T, Mat>::m_weightMatrices.end(),
                                                   weights->begin(),
                                                   weights->end());
        }
        return &this->m_weightMatrices;
	}

    std::vector<MatrixRef>* GetGradients(Datablob< T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_gradientMatrices.clear();
        if (Datablob<T, Mat>* ln0 = _blob->GetBlob("LayerNorm0Data"))
        {
            auto* grads = m_layerNorm0->GetGradients(ln0);
            Layer<T, Mat>::m_gradientMatrices.insert(Layer<T, Mat>::m_gradientMatrices.end(),
                                                     grads->begin(),
                                                     grads->end());
        }
        if (Datablob<T, Mat>* attention = _blob->GetBlob("AttentionLayerData"))
        {
            auto* grads = m_attention->GetGradients(attention);
            Layer<T, Mat>::m_gradientMatrices.insert(Layer<T, Mat>::m_gradientMatrices.end(),
                                                     grads->begin(),
                                                     grads->end());
        }
        if (Datablob<T, Mat>* ln1 = _blob->GetBlob("LayerNorm1Data"))
        {
            auto* grads = m_layerNorm1->GetGradients(ln1);
            Layer<T, Mat>::m_gradientMatrices.insert(Layer<T, Mat>::m_gradientMatrices.end(),
                                                     grads->begin(),
                                                     grads->end());
        }
        if (Datablob<T, Mat>* denseFC = _blob->GetBlob("DenseFCData"))
        {
            auto* grads = m_denseFC->GetGradients(denseFC);
            Layer<T, Mat>::m_gradientMatrices.insert(Layer<T, Mat>::m_gradientMatrices.end(),
                                                     grads->begin(),
                                                     grads->end());
        }
        if (Datablob<T, Mat>* denseProj = _blob->GetBlob("DenseProjData"))
        {
            auto* grads = m_denseProj->GetGradients(denseProj);
            Layer<T, Mat>::m_gradientMatrices.insert(Layer<T, Mat>::m_gradientMatrices.end(),
                                                     grads->begin(),
                                                     grads->end());
        }
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

    virtual std::string GetInputOutputString(Datablob<T, Mat>* _blob) override
    {
        std::string in = _blob->GetMatrix(this->GetInputName())->GetDims().GetString();
        std::string out = _blob->GetMatrix(this->GetOutputName())->GetDims().GetString();

        Datablob<T, Mat>* attentionData     = _blob->GetBlob("AttentionLayerData");
        Datablob<T, Mat>* layerNorm0Data    = _blob->GetBlob("LayerNorm0Data");
        Datablob<T, Mat>* layerNorm1Data    = _blob->GetBlob("LayerNorm1Data");
        Datablob<T, Mat>* denseFCData       = _blob->GetBlob("DenseFCData");
        Datablob<T, Mat>* denseProjData     = _blob->GetBlob("DenseProjData");
        Datablob<T, Mat>* mlpDropoutData    = _blob->GetBlob("MLPDropoutData");

        std::stringstream ss;
        ss << this->GetName() << ": "<<in << " -> " << std::endl;
        ss << "Internal:" << std::endl;
        ss << m_attention->GetInputOutputString(attentionData) << std::endl;
        ss << m_layerNorm0->GetInputOutputString(layerNorm0Data) << std::endl;
        ss << m_layerNorm1->GetInputOutputString(layerNorm1Data) << std::endl;
        ss << m_denseFC->GetInputOutputString(denseFCData) << std::endl;
        ss << m_denseProj->GetInputOutputString(denseProjData) << std::endl;
        ss << m_mlpDropout->GetInputOutputString(mlpDropoutData) << std::endl;
        ss << "=======" << std::endl;
        ss << "-> " << out << std::endl;

        return ss.str();
    }

    void EnsureOutputsAllocated(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        if (!input)
            return;

        const uint32_t  initNEmbed      = _blob->GetUInt("Init_NEmbed");
        const uint32_t  initNHead       = _blob->GetUInt("Init_NHead");
        const uint32_t  initBlockSize   = _blob->GetUInt("Init_BlockSize");
        const float     initDropout     = _blob->GetFloat("Init_Dropout");
        const bool      initForTraining = _blob->GetUInt("TrainingEnabled") > 0u;
        const bool      useBiasDefault  = _blob->GetUInt("UseBias") > 0u;
        const float     residProjScale  = _blob->GetFloat("Init_ResidProjScale");

        this->EnsureMatrix(_blob, "Output_0", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
        this->EnsureMatrix(_blob, "Buffer_0", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
        if (_blob->GetUInt("TrainingEnabled") > 0u)
        {
            this->EnsureMatrix(_blob, "ErrorOut", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
        }

        auto* layerNorm0Data = _blob->GetBlob("LayerNorm0Data");
        if (!layerNorm0Data)
        {
            layerNorm0Data = InitLayerNormBlob<T, Mat>(initForTraining, true, useBiasDefault);
            _blob->Set("LayerNorm0Data", layerNorm0Data);
        }
        layerNorm0Data->Set("Input_0", input);
        m_layerNorm0->EnsureOutputsAllocated(layerNorm0Data);

        auto* layerNorm1Data = _blob->GetBlob("LayerNorm1Data");
        if (!layerNorm1Data)
        {
            layerNorm1Data = InitLayerNormBlob<T, Mat>(initForTraining, true, useBiasDefault);
            _blob->Set("LayerNorm1Data", layerNorm1Data);
        }
        typename MatrixManager<T, Mat>::MatrixRef bufferRef = _blob->AcquireMatrix("Buffer_0");
        Mat* buffer = bufferRef.get();
        layerNorm1Data->Set("Input_0", buffer);
        m_layerNorm1->EnsureOutputsAllocated(layerNorm1Data);

        auto* attentionData = _blob->GetBlob("AttentionLayerData");
        if (!attentionData)
        {
            attentionData = InitCausalSelfAttentionLayerBlob<T, Mat>(initNEmbed, initNHead, initDropout, initBlockSize, initForTraining, useBiasDefault);
            _blob->Set("AttentionLayerData", attentionData);
            if (residProjScale != 1.0f)
            {
                if (Datablob<T, Mat>* projBlob = attentionData ? attentionData->GetBlob("CSA_ProjLayer") : nullptr)
                {
                    if (Mat* projWeights = projBlob->GetMatrix("Dense_Weights"))
                    {
                        Scale(projWeights, projWeights, residProjScale);
                    }
                }
            }
        }
        attentionData->Set("Input_0", layerNorm0Data->GetMatrix("Output_0"));
        m_attention->EnsureOutputsAllocated(attentionData);

        const uint32_t dC = input->GetDimsX();
        const uint32_t dT = input->GetDimsY();
        const uint32_t dB = input->GetDimsZ();
        if (dB != 1u)
        {
            LOG_ERROR() << "TransformerBlockLayer::Forward currently supports batch size 1 only.";
            return;
        }
        const uint32_t batchSize = dT * dB;

        auto* denseFCData = _blob->GetBlob("DenseFCData");
        {
            const uint32_t  outX        = 4u * dC;
            const uint32_t  inX         = dC;
            const bool      needCreate  = !denseFCData;
            bool            needResize  = false;
            if (denseFCData)
            {
                if (Mat* out = denseFCData->GetMatrix("Output_0"))
                    needResize = out->GetDimsX() != outX || out->GetDimsY() != batchSize;
                else
                    needResize = true;
            }
            if (needCreate || needResize)
            {
                if (denseFCData)
                    ReleaseSubblob(denseFCData);
                denseFCData = InitDenseBlob<T, Mat>(outX, inX, batchSize, initForTraining, true, useBiasDefault);
                _blob->Set("DenseFCData", denseFCData);
            }
        }

        auto* denseProjData = _blob->GetBlob("DenseProjData");
        {
            const uint32_t outX = dC;
            const uint32_t inX  = 4u * dC;
            const bool needCreate = !denseProjData;
            bool       needResize = false;
            if (denseProjData)
            {
                if (Mat* out = denseProjData->GetMatrix("Output_0"))
                {
                    needResize = out->GetDimsX() != outX || out->GetDimsY() != batchSize;
                }
                else
                {
                    needResize = true;
                }
            }
            if (needCreate || needResize)
            {
                if (denseProjData)
                {
                    ReleaseSubblob(denseProjData);
                }
                denseProjData = InitDenseBlob<T, Mat>(outX, inX, batchSize, initForTraining, true, useBiasDefault);
                if (residProjScale != 1.0f)
                {
                    if (Mat* mlpProjWeights = denseProjData ? denseProjData->GetMatrix("Dense_Weights") : nullptr)
                    {
                        Scale(mlpProjWeights, mlpProjWeights, residProjScale);
                    }
                }
                _blob->Set("DenseProjData", denseProjData);
            }
        }

        auto* mlpDropoutData = _blob->GetBlob("MLPDropoutData");
        if (!mlpDropoutData)
        {
            mlpDropoutData = InitDropoutBlob<T, Mat>(initDropout, initForTraining);
            _blob->Set("MLPDropoutData", mlpDropoutData);
        }
    }

    void DynamicInit(Datablob<T, Mat>* _blob, Mat* _input)
    {
        // Allocate an output matrix.
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        assert(_input);

        auto output = inst.AllocateMatrix({_input->GetDimsX(), _input->GetDimsY(), _input->GetDimsZ()}, "Output_0");
        _blob->Set("Output_0", output);

        auto buffer = inst.AllocateMatrix({_input->GetDimsX(), _input->GetDimsY(), _input->GetDimsZ()}, "Buffer_0");
        _blob->Set("Buffer_0", buffer);

        if(_blob->GetUInt("TrainingEnabled") > 0u)
        {
            auto outputErr = inst.AllocateMatrix({_input->GetDimsX(), _input->GetDimsY(), _input->GetDimsZ()}, "ErrorOut");
            _blob->Set("ErrorOut", outputErr);
        }
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {
        auto inputRef  = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        auto bufferRef = _blob->AcquireMatrix("Buffer_0");
        Mat* buffer = bufferRef.get();

        auto outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();

        auto* attentionData     = _blob->GetBlob("AttentionLayerData");
        auto* layerNorm0Data    = _blob->GetBlob("LayerNorm0Data");
        auto* layerNorm1Data    = _blob->GetBlob("LayerNorm1Data");
        auto* denseFCData       = _blob->GetBlob("DenseFCData");
        auto* denseProjData     = _blob->GetBlob("DenseProjData");
        auto* mlpDropoutData    = _blob->GetBlob("MLPDropoutData");
        const uint32_t dC = input->GetDimsX();
        const uint32_t dT = input->GetDimsY();
        const uint32_t dB = input->GetDimsZ();

        //x = x + Attention(LN0(x))
        /////////////////////////////
        m_layerNorm0->SetInput(layerNorm0Data, input);
        m_layerNorm0->Run(layerNorm0Data);
         
        m_attention->SetInput(attentionData, m_layerNorm0->GetOutput(layerNorm0Data).get());
        m_attention->Run(attentionData);
        Add(buffer, input, attentionData->GetMatrix("Output_0"));
        
        //x = x + MLP(LN1(x))
        /////////////////////////////
        m_layerNorm1->SetInput(layerNorm1Data, buffer);
        m_layerNorm1->Run(layerNorm1Data);
        
        // Dense FC
        m_denseFC->SetInput(denseFCData, m_layerNorm1->GetOutput(layerNorm1Data).get());
        m_denseFC->Run(denseFCData);
        
        // Dense Proj    
        m_denseProj->SetInput(denseProjData, m_denseFC->GetOutput(denseFCData).get());
        m_denseProj->Run(denseProjData);
        
        // Dropout
        m_mlpDropout->SetInput(mlpDropoutData, m_denseProj->GetOutput(denseProjData).get());
        m_mlpDropout->Run(mlpDropoutData);
        Add(output, buffer, mlpDropoutData->GetMatrix("Output_0"));
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef errorRef     = _blob->AcquireMatrix("ErrorInput_0");
        Mat* error = errorRef.get();
        typename MatrixManager<T, Mat>::MatrixRef errorOutRef  = _blob->AcquireMatrix("ErrorOut");
        Mat* errorOut = errorOutRef.get();
        typename MatrixManager<T, Mat>::MatrixRef lastInputRef = _blob->AcquireMatrix("Input_0");
        Mat* lastInput = lastInputRef.get();

        auto* attentionData  = _blob->GetBlob("AttentionLayerData");
        auto* layerNorm0Data = _blob->GetBlob("LayerNorm0Data");
        auto* layerNorm1Data = _blob->GetBlob("LayerNorm1Data");
        auto* denseFCData    = _blob->GetBlob("DenseFCData");
        auto* denseProjData  = _blob->GetBlob("DenseProjData");
        auto* mlpDropoutData = _blob->GetBlob("MLPDropoutData");

        auto bufferRef = _blob->AcquireMatrix("Buffer_0");
        Mat* buffer = bufferRef.get();
        
        const uint32_t dC = lastInput->GetDimsX();
        const uint32_t dT = lastInput->GetDimsY();
        const uint32_t dB = lastInput->GetDimsZ();
        if (dB != 1u)
        {
            LOG_ERROR() << "TransformerBlockLayer::Backwards currently supports batch size 1 only.";
            return;
        }

        m_mlpDropout->SetErrorInput(mlpDropoutData, errorRef);
        m_mlpDropout->Backwards(mlpDropoutData);
        
        m_denseProj->SetErrorInput(denseProjData, m_mlpDropout->GetOutputError(mlpDropoutData));
        m_denseProj->Backwards(denseProjData);

        m_denseFC->SetErrorInput(denseFCData, m_denseProj->GetOutputError(denseProjData));
        m_denseFC->Backwards(denseFCData);

        m_layerNorm1->SetErrorInput(layerNorm1Data, m_denseFC->GetOutputError(denseFCData));
        m_layerNorm1->Backwards(layerNorm1Data);

        Add(buffer, error, m_layerNorm1->GetOutputError(layerNorm1Data).get());

        m_attention->SetErrorInput(attentionData, bufferRef);
        m_attention->Backwards(attentionData);

        m_layerNorm0->SetErrorInput(layerNorm0Data, m_attention->GetOutputError(attentionData));
        m_layerNorm0->Backwards(layerNorm0Data);

        Add(errorOut, buffer, m_layerNorm0->GetOutputError(layerNorm0Data).get());
    }

    private:
    void ReleaseSubblob(Datablob<T, Mat>* _blob)
    {
        if (!_blob)
            return;
        delete _blob;
    }
    void EnsureDenseBlobSized(Datablob<T, Mat>* _blob, uint32_t _outX, uint32_t _inX, uint32_t _batchSize)
    {
        if (!_blob)
            return;
        this->EnsureMatrix(_blob, "Output_0"                  , _outX, _batchSize);
        this->EnsureMatrix(_blob, "Dense_OutputPreActivation" , _outX, _batchSize);
        if (_blob->GetMatrix("Dense_LinearOut"))
        {
            this->EnsureMatrix(_blob, "Dense_LinearOut", _outX, _batchSize);
            this->EnsureMatrix(_blob, "Dense_Delta"    , _outX, _batchSize);
            this->EnsureMatrix(_blob, "ErrorOut"       , _inX , _batchSize);
            this->EnsureMatrix(_blob, "Dense_LastInput", _inX , _batchSize);
        }
    }

    public:

    virtual void GetSublayerPairs(  std::vector<typename Layer<T,Mat>::sublayerinfo>& _out,
                                    Datablob<T,Mat>*                                  _blob) override
    {
        _out.push_back(typename Layer<T,Mat>::sublayerinfo{"AttentionLayerData"  , m_attention   , _blob->GetBlob("AttentionLayerData")});
        _out.push_back(typename Layer<T,Mat>::sublayerinfo{"LayerNorm0Data"      , m_layerNorm0  , _blob->GetBlob("LayerNorm0Data")});
        _out.push_back(typename Layer<T,Mat>::sublayerinfo{"LayerNorm1Data"      , m_layerNorm1  , _blob->GetBlob("LayerNorm1Data")});
        _out.push_back(typename Layer<T,Mat>::sublayerinfo{"DenseFCData"         , m_denseFC     , _blob->GetBlob("DenseFCData")});
        _out.push_back(typename Layer<T,Mat>::sublayerinfo{"DenseProjData"       , m_denseProj   , _blob->GetBlob("DenseProjData")});
        _out.push_back(typename Layer<T,Mat>::sublayerinfo{"MLPDropoutData"      , m_mlpDropout  , _blob->GetBlob("MLPDropoutData")});
    }

    LayerNorm<T,Mat>*                               m_layerNorm0;
    LayerNorm<T,Mat>*                               m_layerNorm1;
    CausalSelfAttentionLayer<T, Identity<T>, Mat>*  m_attention;
    Dense<T, Gelu<T,Mat>, Mat>*                     m_denseFC;
    Dense<T, Identity<T>, Mat>*                     m_denseProj;
    DropoutLayer<T, Mat>*                           m_mlpDropout;
};
