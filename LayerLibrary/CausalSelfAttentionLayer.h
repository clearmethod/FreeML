#pragma once

#include "Layer.h"
#include <MatrixLibrary/MatrixManager.h>
#include <MatrixLibrary/MatrixBase.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>

#include <MatrixLibrary/CPU/MatrixCPU.h>

#include <LayerLibrary/Dense.h>
#include <LayerLibrary/DropoutLayer.h>

#include <ToolsLibrary/Timer.h>
#include <ToolsLibrary/Tools.h>

#include <cmath>
#include <functional>
#include <utility>
#include <limits>
#include <string>
#include <type_traits>

#include "Datablob.h"

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitCausalSelfAttentionLayerBlob(
                                               uint32_t _n_embed,
                                               uint32_t _n_head,
                                               float    _dropout         = 0.0,
                                               uint32_t _block_size      = 1024,
                                               bool     _initForTraining = true,
                                               bool     _useBias         = false
                                             )
{
    Datablob<T, Mat>* blob           = new Datablob<T, Mat>();
    MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

    blob->Set("TrainingEnabled" , _initForTraining? 1u : 0u);
    blob->Set("NumHeads"        , _n_head);
    blob->Set("Init_NEmbed"     , _n_embed);
    blob->Set("Init_BlockSize"  , _block_size);
    blob->Set("Init_Dropout"    , _dropout);
    blob->Set("UseBias"         , _useBias ? 1u : 0u);

    // Init attention dense data
    Datablob<T, Mat>* csa_attention = InitDenseBlob<T, Mat>(3u * _n_embed, _n_embed, _block_size, _initForTraining, true, _useBias, false);  //3x because Query, Key, Value values output.
    blob->Set("CSA_AttentionLayer", csa_attention);
    
    // Init projection dense data
    Datablob<T, Mat>* csa_proj = InitDenseBlob<T, Mat>(_n_embed, _n_embed, _block_size, _initForTraining, true, _useBias, true);
    blob->Set("CSA_ProjLayer", csa_proj);

    // One full batch/head mask is retained for backward.
    blob->Set("CSA_AttentionDropout", InitDropoutBlob<T,Mat>(_dropout,_initForTraining));

    // Init resid drop out
    Datablob<T, Mat>* csa_Residual_Dropout = InitDropoutBlob<T, Mat>(_dropout, _initForTraining);
    blob->Set("CSA_ResidualDropout", csa_Residual_Dropout);

    // Init resid drop out
    auto mask = inst.AllocateMatrix(MatrixShape::Matrix(_block_size,_block_size), "Mask");
    TriangleMatrix<T>(mask.get(), TriangleDirection::Lower, T(0), -std::numeric_limits<T>::infinity());
    blob->Set("TriangleMatrix", mask);
    auto maskFinite = inst.AllocateMatrix(MatrixShape::Matrix(_block_size,_block_size), "MaskFinite");
    TriangleMatrix<T>(maskFinite.get(), TriangleDirection::Lower, T(1), T(0));
    blob->Set("TriangleFiniteMask", maskFinite);

    const uint32_t headDim = _n_head ? (_n_embed / _n_head) : 0u;
    const T scale = headDim ? (T(1) / std::sqrt(static_cast<T>(headDim))) : T(0);
    auto scalarScale = inst.AllocateMatrix(MatrixShape::Matrix(1u,1u), "ScalarScale");
    scalarScale->SetValue(0u, 0u, scale);
    blob->Set("ScalarScale", scalarScale);
    blob->Set("ScalarScaleHeadDim", headDim);

    return blob;
}

template<class T, class ActFunc, class Mat = MatrixCPU<T>>
class CausalSelfAttentionLayer : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

    public:
    CausalSelfAttentionLayer()
    {
        m_attention         = new Dense<T,ActFunc, Mat, false>();
        m_proj              = new Dense<T,ActFunc, Mat, true>();
        m_attentionDropout  = new DropoutLayer<T, Mat>();
        m_residualDropout   = new DropoutLayer<T, Mat>();
    }
    ~CausalSelfAttentionLayer()
    {
        delete m_attention;
        delete m_proj;
        delete m_attentionDropout;
        delete m_residualDropout;
    }

    virtual std::string GetTypeName() override
    {
        return "CausalSelfAttentionLayer";
    }

    std::vector<MatrixRef>* GetWeights(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_weightMatrices.clear();
        if (Datablob<T, Mat>* attentionBlob = _blob->GetBlob("CSA_AttentionLayer"))
        {
            auto* weights = m_attention->GetWeights(attentionBlob);
            Layer<T, Mat>::m_weightMatrices.insert(Layer<T, Mat>::m_weightMatrices.end(),
                                                   weights->begin(),
                                                   weights->end());
        }
        if (Datablob<T, Mat>* projBlob = _blob->GetBlob("CSA_ProjLayer"))
        {
            auto* weights = m_proj->GetWeights(projBlob);
            Layer<T, Mat>::m_weightMatrices.insert(Layer<T, Mat>::m_weightMatrices.end(),
                                                   weights->begin(),
                                                   weights->end());
        }
        return &this->m_weightMatrices;
	}

    std::vector<MatrixRef>* GetGradients(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_gradientMatrices.clear();
        if (Datablob<T, Mat>* attentionBlob = _blob->GetBlob("CSA_AttentionLayer"))
        {
            auto* grads = m_attention->GetGradients(attentionBlob);
            Layer<T, Mat>::m_gradientMatrices.insert(Layer<T, Mat>::m_gradientMatrices.end(),
                                                     grads->begin(),
                                                     grads->end());
        }
        if (Datablob<T, Mat>* projBlob = _blob->GetBlob("CSA_ProjLayer"))
        {
            auto* grads = m_proj->GetGradients(projBlob);
            Layer<T, Mat>::m_gradientMatrices.insert(Layer<T, Mat>::m_gradientMatrices.end(),
                                                     grads->begin(),
                                                     grads->end());
        }
		return &this->m_gradientMatrices;
    }

    uint32_t GetOutputErrorCount() override
    {
        return 1u;
    }

    MatrixRef GetOutputError(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        return _blob->AcquireMatrix("ErrorOutput_0");
    }

    MatrixRef GetOutput(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        return _blob->AcquireMatrix("Output_0");
    }

    void EnsureAttentionLayer(uint32_t _dC, uint32_t _dTB, bool _initForTraining, bool _useBias, Datablob<T,Mat>* _blob)
    {
        Datablob<T, Mat>* attentionBlob = _blob->GetBlob("CSA_AttentionLayer");
        {
            const uint32_t  outX = 3u * _dC;
            const uint32_t  inX = _dC;
            const bool      needCreate = !attentionBlob;
            bool            needResize = false;
            if (attentionBlob)
            {
                if (Mat* out = attentionBlob->GetMatrix("Output_0"))
                    needResize = !attentionBlob->GetMatrix("Dense_Weights") || attentionBlob->GetMatrix("Dense_Weights")->GetShape()!=MatrixShape::Matrix(outX,inX);
                else
                    needResize = true;
            }
            if (needCreate || needResize)
            {
                if (attentionBlob)
                    ReleaseSubblob(attentionBlob);
                attentionBlob = InitDenseBlob<T, Mat>(outX, inX, _dTB, _initForTraining, true, _useBias, false);
                _blob->Set("CSA_AttentionLayer", attentionBlob);
            }
        }
    }

    void EnsureOutputsAllocated(Datablob<T,Mat>* blob) override
    {
        Mat* input=blob->GetMatrix("Input_0");if(!input) return;
        const auto dims=GetShape(blob,input);
        const bool training=blob->GetUInt("TrainingEnabled")!=0u;
        this->EnsureMatrix(blob,"Output_0",input->GetShape());
        if(training) this->EnsureMatrix(blob,"ErrorOutput_0",input->GetShape());
        const auto sequence=input->GetShape();
        const auto scores=MatrixShape::Image(dims.B_batch,dims.nHead,dims.T_block_size,dims.T_block_size);
        const auto heads=MatrixShape::Image(dims.B_batch,dims.nHead,dims.headDim,dims.T_block_size);
        for(const char* key : {"Scratch_Q","Scratch_K","Scratch_V","Scratch_Y"})
            this->EnsureMatrix(blob,key,sequence);
        for(const char* key : {"AttentionProbsAll","AttentionDroppedAll","Scratch_dScores"})
            this->EnsureMatrix(blob,key,scores);
        if(training) {
            this->EnsureMatrix(blob,"Scratch_dAtt",scores);
            for(const char* key : {"Scratch_dQh","Scratch_dKh","Scratch_dVh"}) this->EnsureMatrix(blob,key,heads);
            this->EnsureMatrix(blob,"Scratch_QKVGrad",MatrixShape::Sequence(dims.B_batch,3u*dims.C_EmbedDims,dims.T_block_size));
        }
        const auto maskShape=MatrixShape::Matrix(dims.T_block_size,dims.T_block_size);
        Mat* mask=blob->GetMatrix("TriangleMatrix");
        if(!mask || mask->GetShape()!=maskShape) {
            this->EnsureMatrix(blob,"TriangleMatrix",maskShape);
            TriangleMatrix(blob->GetMatrix("TriangleMatrix"),TriangleDirection::Lower,T(0),-std::numeric_limits<T>::infinity());
        }
        if(blob->GetUInt("ScalarScaleHeadDim")!=dims.headDim) {
            blob->GetMatrix("ScalarScale")->SetValue(0u,0u,T(1)/std::sqrt(T(dims.headDim)));
            blob->Set("ScalarScaleHeadDim",dims.headDim);
        }
        // Dense resizes activations without recreating shared parameters.
        EnsureAttentionLayer(dims.C_EmbedDims,dims.TB,training,blob->GetUInt("UseBias")!=0u,blob);
        if(!blob->GetBlob("CSA_ProjLayer"))
            blob->Set("CSA_ProjLayer",InitDenseBlob<T,Mat>(dims.C_EmbedDims,dims.C_EmbedDims,dims.TB,training,true,blob->GetUInt("UseBias")!=0u,true));
        if(!blob->GetBlob("CSA_AttentionDropout"))
            blob->Set("CSA_AttentionDropout",InitDropoutBlob<T,Mat>(blob->GetFloat("Init_Dropout"),training));
    }

    void DynamicInit(Datablob<T, Mat>* _blob, Mat* _input)
    {
        // Allocate an output matrix.
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        assert(_input);

        auto output = inst.AllocateMatrix(_input->GetShape(), "Output_0");
        _blob->Set("Output_0", output);

        if(_blob->GetUInt("TrainingEnabled") > 0u)
        {
            auto outputErr = inst.AllocateMatrix(_input->GetShape(), "ErrorOutput_0");
            _blob->Set("ErrorOutput_0", outputErr);
        }
    }

    void Forward(Datablob<T,Mat>* blob) override
    {
        EnsureOutputsAllocated(blob);
        Mat* input=blob->GetMatrix("Input_0");const auto dims=GetShape(blob,input);
        auto* attention=blob->GetBlob("CSA_AttentionLayer");
        m_attention->SetInput(attention,input);m_attention->Forward(attention);
        SplitQKV(blob->GetMatrix("Scratch_Q"),blob->GetMatrix("Scratch_K"),blob->GetMatrix("Scratch_V"),attention->GetMatrix("Output_0"));
        const auto heads=MatrixShape::Image(dims.B_batch,dims.nHead,dims.headDim,dims.T_block_size);
        auto q=this->CachedView(blob,"Scratch_QHeads",blob->AcquireMatrix("Scratch_Q"),heads);
        auto k=this->CachedView(blob,"Scratch_KHeads",blob->AcquireMatrix("Scratch_K"),heads);
        auto v=this->CachedView(blob,"Scratch_VHeads",blob->AcquireMatrix("Scratch_V"),heads);
        auto y=this->CachedView(blob,"Scratch_YHeads",blob->AcquireMatrix("Scratch_Y"),heads);
        const MatrixStrides features{dims.C_EmbedDims,1u,dims.headDim,dims.T_block_size*dims.C_EmbedDims};
        const MatrixStrides scores{dims.T_block_size,1u,dims.T_block_size*dims.T_block_size};
        Mat* score=blob->GetMatrix("Scratch_dScores");
        MatMul_Batched<TransposeMode::Right>(score,q.get(),k.get(),scores,features,features);
        Scale(score,score,blob->GetMatrix("ScalarScale"));
        Add(score,score,blob->GetMatrix("TriangleMatrix"));
        Mat* probabilities=blob->GetMatrix("AttentionProbsAll");Softmax(probabilities,score);
        auto* dropout=blob->GetBlob("CSA_AttentionDropout");
        m_attentionDropout->SetInput(dropout,probabilities);m_attentionDropout->Run(dropout);
        Copy(blob->GetMatrix("AttentionDroppedAll"),dropout->GetMatrix("Output_0"));
        MatMul_Batched<TransposeMode::None>(y.get(),blob->GetMatrix("AttentionDroppedAll"),v.get(),features,scores,features);
        auto* projection=blob->GetBlob("CSA_ProjLayer");
        m_proj->SetInput(projection,blob->GetMatrix("Scratch_Y"));m_proj->Forward(projection);
        auto* residual=blob->GetBlob("CSA_ResidualDropout");
        m_residualDropout->SetInput(residual,projection->GetMatrix("Output_0"));m_residualDropout->Run(residual);
        Copy(blob->GetMatrix("Output_0"),residual->GetMatrix("Output_0"));
    }

    void Backwards(Datablob<T,Mat>* blob) override
    {
        const auto dims=GetShape(blob,blob->GetMatrix("Input_0"));
        auto* residual=blob->GetBlob("CSA_ResidualDropout");
        m_residualDropout->SetErrorInput(residual,blob->GetMatrix("ErrorInput_0"));m_residualDropout->Backwards(residual);
        auto* projection=blob->GetBlob("CSA_ProjLayer");
        m_proj->SetErrorInput(projection,residual->GetMatrix("ErrorOutput_0"));m_proj->Backwards(projection);
        const auto heads=MatrixShape::Image(dims.B_batch,dims.nHead,dims.headDim,dims.T_block_size);
        auto q=this->CachedView(blob,"Scratch_QHeads",blob->AcquireMatrix("Scratch_Q"),heads);
        auto k=this->CachedView(blob,"Scratch_KHeads",blob->AcquireMatrix("Scratch_K"),heads);
        auto v=this->CachedView(blob,"Scratch_VHeads",blob->AcquireMatrix("Scratch_V"),heads);
        auto dy=this->CachedView(blob,"GradientHeads",projection->AcquireMatrix("ErrorOutput_0"),heads);
        const MatrixStrides features{dims.C_EmbedDims,1u,dims.headDim,dims.T_block_size*dims.C_EmbedDims};
        const MatrixStrides scores{dims.T_block_size,1u,dims.T_block_size*dims.T_block_size};
        const MatrixStrides gradients{dims.headDim,1u,dims.T_block_size*dims.headDim};
        Mat* dAttention=blob->GetMatrix("Scratch_dAtt");
        MatMul_Batched<TransposeMode::Right>(dAttention,dy.get(),v.get(),scores,features,features);
        MatMul_Batched<TransposeMode::Left>(blob->GetMatrix("Scratch_dVh"),blob->GetMatrix("AttentionDroppedAll"),dy.get(),gradients,scores,features);
        auto* dropout=blob->GetBlob("CSA_AttentionDropout");
        m_attentionDropout->SetErrorInput(dropout,dAttention);m_attentionDropout->Backwards(dropout);
        Mat* dScores=blob->GetMatrix("Scratch_dScores");
        AttentionScoresBackwards(dScores,blob->GetMatrix("AttentionProbsAll"),dropout->GetMatrix("ErrorOutput_0"),blob->GetMatrix("ScalarScale"));
        MatMul_Batched<TransposeMode::None>(blob->GetMatrix("Scratch_dQh"),dScores,k.get(),gradients,scores,features);
        MatMul_Batched<TransposeMode::Left>(blob->GetMatrix("Scratch_dKh"),dScores,q.get(),gradients,scores,features);
        MergeQKV(blob->GetMatrix("Scratch_QKVGrad"),blob->GetMatrix("Scratch_dQh"),blob->GetMatrix("Scratch_dKh"),blob->GetMatrix("Scratch_dVh"));
        auto* attention=blob->GetBlob("CSA_AttentionLayer");
        m_attention->SetErrorInput(attention,blob->GetMatrix("Scratch_QKVGrad"));m_attention->Backwards(attention);
        Copy(blob->GetMatrix("ErrorOutput_0"),attention->GetMatrix("ErrorOutput_0"));
    }

    struct AttentionDims
    {
        uint32_t C_EmbedDims = 0u;
        uint32_t T_block_size = 0u;
        uint32_t B_batch = 0u;
        uint32_t TB = 0u;
        uint32_t nHead = 0u;
        uint32_t headDim = 0u;
    };

    AttentionDims GetShape(Datablob<T, Mat>* _blob, Mat* _input) const
    {
        AttentionDims dims{};
        dims.C_EmbedDims = _input->GetWidth();
        dims.T_block_size = _input->GetHeight();
        dims.B_batch = _input->GetBatchCount();
        if(_input->GetChannelCount()!=1u) throw std::invalid_argument("Attention expects sequence tensors.");
        dims.TB = dims.T_block_size * dims.B_batch;
        dims.nHead = _blob->GetUInt("NumHeads");
        assert(dims.nHead > 0u);
        assert(dims.C_EmbedDims % dims.nHead == 0u);
        dims.headDim = dims.C_EmbedDims / dims.nHead;
        return dims;
    }

    void ReleaseSubblob(Datablob<T, Mat>* _blob)
    {
        if (!_blob)
        {
            return;
        }
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        const auto& handles = _blob->GetAllMatrixRefData();
        for (const auto& kv : handles)
        {
            const std::string& key = kv.first;
            if (key.rfind("Input_", 0) == 0 || key.rfind("ErrorInput_", 0) == 0)
            {
                continue;
            }
            inst.RemoveMatrix(kv.second.get());
        }
        delete _blob;
    }

    virtual void GetSublayerPairs(std::vector<typename Layer<T,Mat>::sublayerinfo>& _out,
                          Datablob<T,Mat>* _blob) override
    {
        _out.push_back(typename Layer<T,Mat>::sublayerinfo{"CSA_AttentionLayer" , m_attention        , _blob->GetBlob("CSA_AttentionLayer")});
        _out.push_back(typename Layer<T,Mat>::sublayerinfo{"CSA_ProjLayer"      , m_proj             , _blob->GetBlob("CSA_ProjLayer")});
        _out.push_back(typename Layer<T,Mat>::sublayerinfo{"CSA_ResidualDropout", m_residualDropout  , _blob->GetBlob("CSA_ResidualDropout")});

        _out.push_back(typename Layer<T,Mat>::sublayerinfo{"CSA_AttentionDropout",m_attentionDropout,_blob->GetBlob("CSA_AttentionDropout")});
    }

    private:
    public:
    // Member layers
    Dense<T,ActFunc,Mat, false>* m_attention        = nullptr;
    Dense<T,ActFunc,Mat, true>*  m_proj             = nullptr;
    DropoutLayer<T,Mat>*         m_attentionDropout = nullptr;
    DropoutLayer<T,Mat>*         m_residualDropout  = nullptr;
};
