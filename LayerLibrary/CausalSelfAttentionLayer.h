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

    // Init attention drop out per head (needed to preserve masks for backward pass)
    for (uint32_t head = 0; head < _n_head; ++head)
    {
        Datablob<T, Mat>* csa_Attention_Dropout = InitDropoutBlob<T, Mat>(_dropout, _initForTraining);
        blob->Set("CSA_AttentionDropout_" + std::to_string(head), csa_Attention_Dropout);
    }

    // Init resid drop out
    Datablob<T, Mat>* csa_Residual_Dropout = InitDropoutBlob<T, Mat>(_dropout, _initForTraining);
    blob->Set("CSA_ResidualDropout", csa_Residual_Dropout);

    // Init resid drop out
    auto mask = inst.AllocateMatrix({_block_size, _block_size}, "Mask");
    TriangleMatrix<T>(mask.get(), TriangleDirection::Lower, T(0), -std::numeric_limits<T>::infinity());
    blob->Set("TriangleMatrix", mask);
    auto maskFinite = inst.AllocateMatrix({_block_size, _block_size}, "MaskFinite");
    TriangleMatrix<T>(maskFinite.get(), TriangleDirection::Lower, T(1), T(0));
    blob->Set("TriangleFiniteMask", maskFinite);

    const uint32_t headDim = _n_head ? (_n_embed / _n_head) : 0u;
    const T scale = headDim ? (T(1) / std::sqrt(static_cast<T>(headDim))) : T(0);
    auto scalarScale = inst.AllocateMatrix({1u, 1u, 1u}, "ScalarScale");
    scalarScale->SetValue(0u, 0u, scale);
    blob->Set("ScalarScale", scalarScale);

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
                    needResize = out->GetDimsX() != outX || out->GetDimsY() != _dTB;
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

    void EnsureOutputsAllocated(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        if (!input)
        {
            return;
        }

        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        const bool initForTraining = _blob->GetUInt("TrainingEnabled") > 0u;
        const bool useBias = _blob->GetUInt("UseBias") > 0u;
        const AttentionDims dims = GetDims(_blob, input);
        const uint32_t attRows = dims.T_block_size;

        // Output / error buffers.
        {
            typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
            Mat* output = outputRef.get();
            const bool outputMismatch = !output
                || output->GetDimsX() != input->GetDimsX()
                || output->GetDimsY() != input->GetDimsY()
                || output->GetDimsZ() != input->GetDimsZ();
            if (outputMismatch)
            {
                if (output)
                {
                    inst.RemoveMatrix(output);
                }
                auto outputRefNew = inst.AllocateMatrix({input->GetDimsX(), input->GetDimsY(), input->GetDimsZ()}, "Output_0");
                _blob->Set("Output_0", outputRefNew);
            }
        }

        if (initForTraining)
        {
            typename MatrixManager<T, Mat>::MatrixRef outputErrRef = _blob->AcquireMatrix("ErrorOut");
            Mat* outputErr = outputErrRef.get();
            const bool errMismatch = !outputErr
                || outputErr->GetDimsX() != input->GetDimsX()
                || outputErr->GetDimsY() != input->GetDimsY()
                || outputErr->GetDimsZ() != input->GetDimsZ();
            if (errMismatch)
            {
                if (outputErr)
                {
                    inst.RemoveMatrix(outputErr);
                }
                auto outputErrRefNew = inst.AllocateMatrix({input->GetDimsX(), input->GetDimsY(), input->GetDimsZ()}, "ErrorOut");
                _blob->Set("ErrorOut", outputErrRefNew);
            }
        }

        // Attention projection (QKV) and output projection subblobs.
        EnsureAttentionLayer(dims.C_EmbedDims, dims.TB, initForTraining, useBias, _blob);
        Datablob<T, Mat>* attentionBlob = _blob->GetBlob("CSA_AttentionLayer");

        Datablob<T, Mat>* projBlob = _blob->GetBlob("CSA_ProjLayer");
        {
            const uint32_t outX = dims.C_EmbedDims;
            const uint32_t inX = dims.C_EmbedDims;
            const bool needCreate = !projBlob;
            bool needResize = false;
            if (projBlob)
            {
                if (Mat* out = projBlob->GetMatrix("Output_0"))
                {
                    needResize = out->GetDimsX() != outX || out->GetDimsY() != dims.TB;
                }
                else
                {
                    needResize = true;
                }
            }
            if (needCreate || needResize)
            {
                if (projBlob)
                {
                    ReleaseSubblob(projBlob);
                }
                projBlob = InitDenseBlob<T, Mat>(outX, inX, dims.TB, initForTraining, true, useBias, true);
                _blob->Set("CSA_ProjLayer", projBlob);
            }
        }

        // Per-head attention outputs and dropout buffers.
        for (uint32_t head = 0; head < dims.nHead; ++head)
        {
            Datablob<T, Mat>* attentionDropoutBlob = GetAttentionDropoutBlob(_blob, head);
            if (attentionDropoutBlob)
            {
                this->EnsureMatrix(attentionDropoutBlob, "Output_0", attRows, attRows, dims.B_batch);
                if (attentionDropoutBlob->GetUInt("TrainingEnabled") > 0u)
                {
                    this->EnsureMatrix(attentionDropoutBlob, "Mask", attRows, attRows, dims.B_batch);
                    this->EnsureMatrix(attentionDropoutBlob, "ErrorOut", attRows, attRows, dims.B_batch);
                }
            }

            const std::string attKey = HeadAttKey(head);
            this->EnsureMatrix(_blob, attKey.c_str(), attRows, attRows, dims.B_batch);
        }

        // Residual dropout buffers.
        Datablob<T, Mat>* residualDropoutBlob = _blob->GetBlob("CSA_ResidualDropout");
        if (residualDropoutBlob)
        {
            this->EnsureMatrix(residualDropoutBlob, "Output_0", dims.C_EmbedDims, dims.T_block_size, dims.B_batch);
            if (residualDropoutBlob->GetUInt("TrainingEnabled") > 0u)
            {
                this->EnsureMatrix(residualDropoutBlob, "Mask", dims.C_EmbedDims, dims.T_block_size, dims.B_batch);
                this->EnsureMatrix(residualDropoutBlob, "ErrorOut", dims.C_EmbedDims, dims.T_block_size, dims.B_batch);
            }
        }

        // Scratch buffers.
        this->EnsureMatrix(_blob, "Scratch_Q"       , dims.C_EmbedDims     , dims.T_block_size     , dims.B_batch);
        this->EnsureMatrix(_blob, "Scratch_K"       , dims.C_EmbedDims     , dims.T_block_size     , dims.B_batch);
        this->EnsureMatrix(_blob, "Scratch_V"       , dims.C_EmbedDims     , dims.T_block_size     , dims.B_batch);
        this->EnsureMatrix(_blob, "Scratch_Y"       , dims.C_EmbedDims     , dims.T_block_size     , dims.B_batch);
        this->EnsureMatrix(_blob, "Scratch_dY"      , dims.C_EmbedDims     , dims.T_block_size     , dims.B_batch);
        this->EnsureMatrix(_blob, "Scratch_dAtt"    , attRows    , attRows    , dims.nHead);
        this->EnsureMatrix(_blob, "Scratch_dScores" , attRows    , attRows    , dims.nHead);
        this->EnsureMatrix(_blob, "Scratch_dQh"     , dims.headDim, dims.T_block_size    , dims.nHead);
        this->EnsureMatrix(_blob, "Scratch_dKh"     , dims.headDim, dims.T_block_size    , dims.nHead);
        this->EnsureMatrix(_blob, "Scratch_dVh"     , dims.headDim, dims.T_block_size    , dims.nHead);
        this->EnsureMatrix(_blob, "Scratch_QKVGrad" , 3u * dims.C_EmbedDims, dims.T_block_size     , dims.B_batch);

        if (attentionBlob)
        {
            typename MatrixManager<T, Mat>::MatrixRef qkvGradRef = _blob->AcquireMatrix("Scratch_QKVGrad");
            Mat* qkvGrad = qkvGradRef.get();
            typename MatrixManager<T, Mat>::MatrixRef attErrInputRef = attentionBlob->AcquireMatrix("ErrorInput_0");
            if (qkvGrad && !attErrInputRef.get())
            {
                attentionBlob->Set("ErrorInput_0", qkvGrad);
            }

            typename MatrixManager<T, Mat>::MatrixRef attInputRef = attentionBlob->AcquireMatrix("Input_0");
            if (!attInputRef.get())
            {
                attentionBlob->Set("Input_0", input);
            }
        }

        if(!_blob->GetMatrix("TriangleMatrix"))
        {
            this->EnsureMatrix(_blob, "TriangleMatrix", attRows, attRows);
            typename MatrixManager<T, Mat>::MatrixRef triMaskRef = _blob->AcquireMatrix("TriangleMatrix");
            Mat* triMask = triMaskRef.get();
            if (triMask)
            {
                TriangleMatrix<T>(triMask, TriangleDirection::Lower, T(0), -std::numeric_limits<T>::infinity());
            }
        }
        if (!_blob->GetMatrix("TriangleFiniteMask"))
        {
            this->EnsureMatrix(_blob, "TriangleFiniteMask", attRows, attRows);
            typename MatrixManager<T, Mat>::MatrixRef triFiniteMaskRef = _blob->AcquireMatrix("TriangleFiniteMask");
            Mat* triFiniteMask = triFiniteMaskRef.get();
            if (triFiniteMask)
            {
                TriangleMatrix<T>(triFiniteMask, TriangleDirection::Lower, T(1), T(0));
            }
        }
    }

    void DynamicInit(Datablob<T, Mat>* _blob, Mat* _input)
    {
        // Allocate an output matrix.
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        assert(_input);

        auto output = inst.AllocateMatrix({_input->GetDimsX(), _input->GetDimsY(), _input->GetDimsZ()}, "Output_0");
        _blob->Set("Output_0", output);

        if(_blob->GetUInt("TrainingEnabled") > 0u)
        {
            auto outputErr = inst.AllocateMatrix({_input->GetDimsX(), _input->GetDimsY(), _input->GetDimsZ()}, "ErrorOut");
            _blob->Set("ErrorOut", outputErr);
        }
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {
        // 1) Fetch input/output buffers and interpret dims as (C_embeding, T_block, B_batch) in row-major layout.
        auto inputRef   = _blob->AcquireMatrix("Input_0");
        Mat* input      = inputRef.get();
        auto outputRef  = _blob->AcquireMatrix("Output_0");
        Mat* output     = outputRef.get();

        // Query, Key and Value are all size (C,C) where C is the model dimensions (sometimes called embed_dimesions).
        auto qRef = _blob->AcquireMatrix("Scratch_Q");
        auto kRef = _blob->AcquireMatrix("Scratch_K");
        auto vRef = _blob->AcquireMatrix("Scratch_V");
        Mat* Query  = qRef.get();
        Mat* Key    = kRef.get();
        Mat* Value  = vRef.get();

        auto triMaskRef = _blob->AcquireMatrix("TriangleMatrix");
        Mat* triMask    = triMaskRef.get();

        auto scalarScaleRef = _blob->AcquireMatrix("ScalarScale");
        Mat* scalarScale    = scalarScaleRef.get();
        
        auto attScoresAllRef = _blob->AcquireMatrix("Scratch_dScores");
        Mat* attScoresAll    = attScoresAllRef.get();

        auto yCombinedRef   = _blob->AcquireMatrix("Scratch_Y");
        Mat* yCombined      = yCombinedRef.get();

        Datablob<T, Mat>* m_attentionBlob = _blob->GetBlob("CSA_AttentionLayer");
        auto attentionOutRef = m_attentionBlob->AcquireMatrix("Output_0");

        Datablob<T, Mat>* residualDropoutBlob = _blob->GetBlob("CSA_ResidualDropout");
        auto residOutRef = residualDropoutBlob->AcquireMatrix("Output_0");
        Mat* residOut = residOutRef.get();

        Datablob<T, Mat>* projBlob = _blob->GetBlob("CSA_ProjLayer");
        auto projOutRef = projBlob->AcquireMatrix("Output_0");

        const AttentionDims dims    = GetDims(_blob, input);
        const uint32_t      nHead   = dims.nHead;
        const uint32_t      dT      = dims.T_block_size;
        const uint32_t      dB      = dims.B_batch;
        const uint32_t      dC      = dims.C_EmbedDims;
        const uint32_t      headDim = dims.headDim;

        // 2) Project input into packed QKV (3C) using the attention dense layer.
        Mat* attentionInput = input;
        m_attentionBlob->Set("Input_0", attentionInput);
        m_attention->Forward(m_attentionBlob);
        
        // 3) Split packed QKV into separate buffers. Each row is [Query | Key | Value] in row-major order.
        SplitQKV(Query, Key, Value, attentionOutRef.get());

        for (uint32_t head = 0; head < nHead; ++head)
        {
            Mat* attProbsAll = _blob->AcquireMatrix(HeadAttKey(head).c_str()).get();
            for (uint32_t batch = 0; batch < dB; ++batch)
            {
                // Get each head for Query/Key/Value in this batch.
                Mat QueryBatch;
                Query->GetSliceZ(&QueryBatch, batch);
                Mat QueryHead;
                QueryHead.SetData(QueryBatch.GetData(), QueryBatch.GetOffset() + head * headDim);
                QueryHead.SetDims({ headDim, dT });

                Mat KeyBatch;
                Key->GetSliceZ(&KeyBatch, batch);
                Mat KeyHead;
                KeyHead.SetData(KeyBatch.GetData(), KeyBatch.GetOffset() + head * headDim);
                KeyHead.SetDims({ headDim, dT });

                // 4a) Compute attention scores for this head and batch.
                Mat attScores;
                attScoresAll->GetSliceZ(&attScores, head);
                MatMul_Strided<TransposeMode::Right>(&attScores,
                                                     &QueryHead,
                                                     &KeyHead,
                                                     dC, 1u,
                                                     dC, 1u,
                                                     dT, 1u);

                // 4b) Compute probability scores for this head and batch.
                ScaleAdd(&attScores, &attScores, scalarScale, triMask);

                // 5) Convert to probabilities.
                Mat attProbs;
                attProbsAll->GetSliceZ(&attProbs, batch);
                Softmax(&attProbs, &attScores);
            }

            // 6) Dropout on attention weights (per-head), if set.
            Mat* attDroppedAll = attProbsAll;
            Datablob<T, Mat>* attentionDropoutBlob = GetAttentionDropoutBlob(_blob, head);
            if (attentionDropoutBlob)
            {
                m_attentionDropout->SetInput(attentionDropoutBlob, attProbsAll);
                m_attentionDropout->Forward(attentionDropoutBlob);
                Mat* dropped = m_attentionDropout->GetOutput(attentionDropoutBlob).get();
                if (dropped)
                {
                    attDroppedAll = dropped;
                }
            }

            for (uint32_t batch = 0; batch < dB; ++batch)
            {
                Mat ValueBatch;
                Value->GetSliceZ(&ValueBatch, batch);
                Mat ValueHead;
                ValueHead.SetData(ValueBatch.GetData(), ValueBatch.GetOffset() + head * headDim);
                ValueHead.SetDims({ headDim, dT });

                Mat yBatch;
                yCombined->GetSliceZ(&yBatch, batch);
                Mat YPerHeadOutput;
                YPerHeadOutput.SetData(yBatch.GetData(), yBatch.GetOffset() + head * headDim);
                YPerHeadOutput.SetDims({ headDim, dT });

                Mat attDropped;
                attDroppedAll->GetSliceZ(&attDropped, batch);
                // 7) Weighted sum of values: y = att * ValueHead.
                MatMul_Strided<TransposeMode::None>(&YPerHeadOutput,
                                                    &attDropped,
                                                    &ValueHead,
                                                    dT, 1u,
                                                    dC, 1u,
                                                    dC, 1u);
            }
        }

        // 8) Output projection back to embedding size, then residual dropout.
        m_proj->SetInput(projBlob, yCombined);
        m_proj->Forward(projBlob);

        // 9) Final drop out and copy to output.
        if (residualDropoutBlob && residualDropoutBlob->GetFloat("Probability") > 0.f)
        {
            m_residualDropout->SetInput(residualDropoutBlob, projOutRef.get());
            m_residualDropout->Forward(residualDropoutBlob);
            Copy(output, residOut);
        }
        else
            Copy(output, m_proj->GetOutput(projBlob).get());
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        // Get all the variables together.
        auto errorRef      = _blob->AcquireMatrix("ErrorInput_0");
        Mat* error         = errorRef.get();
        auto errorOutRef   = _blob->AcquireMatrix("ErrorOut");
        Mat* errorOut      = errorOutRef.get();
        auto lastInputRef  = _blob->AcquireMatrix("Input_0");
        Mat* lastInput     = lastInputRef.get();

        const AttentionDims dims = GetDims(_blob, lastInput);
        const uint32_t nHead     = dims.nHead;
        const uint32_t dT        = dims.T_block_size;
        const uint32_t dC        = dims.C_EmbedDims;
        const uint32_t headDim   = dims.headDim;
        auto scalarScaleRef      = _blob->AcquireMatrix("ScalarScale");
        Mat* scalarScale         = scalarScaleRef.get();
        assert(scalarScale);

        auto vRef          = _blob->AcquireMatrix("Scratch_V");
        auto dAttAllRef    = _blob->AcquireMatrix("Scratch_dAtt");
        auto dScoresAllRef = _blob->AcquireMatrix("Scratch_dScores");
        auto qRef          = _blob->AcquireMatrix("Scratch_Q");
        auto kRef          = _blob->AcquireMatrix("Scratch_K");
        auto qkvGradRef    = _blob->AcquireMatrix("Scratch_QKVGrad");
        auto dQhAllRef     = _blob->AcquireMatrix("Scratch_dQh");
        auto dKhAllRef     = _blob->AcquireMatrix("Scratch_dKh");
        auto dVhAllRef     = _blob->AcquireMatrix("Scratch_dVh");

        Mat* V          = vRef.get();
        Mat* dAttAll    = dAttAllRef.get();
        Mat* dScoresAll = dScoresAllRef.get();
        Mat* Q          = qRef.get();
        Mat* K          = kRef.get();
        Mat* qkvGrad    = qkvGradRef.get();
        Mat* dQhAll     = dQhAllRef.get();
        Mat* dKhAll     = dKhAllRef.get();
        Mat* dVhAll     = dVhAllRef.get();

        // 9) Drop out backwards.
        Datablob<T, Mat>* residualDropoutBlob = _blob->GetBlob("CSA_ResidualDropout");
        Mat* projErrorIn = error;
        if (residualDropoutBlob && residualDropoutBlob->GetFloat("Probability") > 0.f)
        {
            m_residualDropout->SetErrorInput(residualDropoutBlob, error);
            m_residualDropout->Backwards(residualDropoutBlob);
            projErrorIn = m_residualDropout->GetOutputError(residualDropoutBlob).get();
        }

        // 8) Projection backwards.
        Datablob<T, Mat>* projBlob = _blob->GetBlob("CSA_ProjLayer");
        auto yRef = _blob->AcquireMatrix("Scratch_Y");
        auto dYRef = _blob->AcquireMatrix("Scratch_dY");
        Mat* dY = dYRef.get();

        m_proj->SetInput(projBlob, yRef);
        m_proj->SetErrorInput(projBlob, projErrorIn);
        m_proj->Backwards(projBlob);
        Copy(dY, m_proj->GetOutputError(projBlob).get());

        auto triFiniteMaskRef = _blob->AcquireMatrix("TriangleFiniteMask");
        Mat* triFiniteMask    = triFiniteMaskRef.get();

        for (uint32_t head = 0; head < nHead; ++head)
        {
            auto attHeadAllRef = _blob->AcquireMatrix(HeadAttKey(head).c_str());
            Mat* attHeadAll = attHeadAllRef.get();

            Datablob<T, Mat>* attentionDropoutBlob = GetAttentionDropoutBlob(_blob, head);
            Mat* attDroppedAll = attHeadAll;
            if (attentionDropoutBlob)
            {
                attDroppedAll = m_attentionDropout->GetOutput(attentionDropoutBlob).get();
                if (!attDroppedAll)
                    attDroppedAll = attentionDropoutBlob->GetMatrix("Output_0");
            }

            Mat dAtt;
            dAttAll->GetSliceZ(&dAtt, head);
            Mat dScores;
            dScoresAll->GetSliceZ(&dScores, head);
            Mat dQh;
            dQhAll->GetSliceZ(&dQh, head);
            Mat dKh;
            dKhAll->GetSliceZ(&dKh, head);
            Mat dVh;
            dVhAll->GetSliceZ(&dVh, head);

            Mat QueryHead;
            QueryHead.SetData(Q->GetData(), Q->GetOffset() + head * headDim);
            QueryHead.SetDims({ headDim, dT });

            Mat KeyHead;
            KeyHead.SetData(K->GetData(), K->GetOffset() + head * headDim);
            KeyHead.SetDims({ headDim, dT });

            Mat ValueHead;
            ValueHead.SetData(V->GetData(), V->GetOffset() + head * headDim);
            ValueHead.SetDims({ headDim, dT });

            Mat YPerHeadOutput;
            YPerHeadOutput.SetData(dY->GetData(), dY->GetOffset() + head * headDim);
            YPerHeadOutput.SetDims({ headDim, dT });

            // 7)
            MatMul_Strided<TransposeMode::Right>(&dAtt,
                                                    &YPerHeadOutput,
                                                    &ValueHead,
                                                    dC,1u,
                                                    dC,1u,
                                                    dT,1u);

            MatMul_Strided<TransposeMode::Left>( &dVh,
                                                    attDroppedAll,
                                                    &YPerHeadOutput,
                                                    dT,1u,
                                                    dC,1u,
                                                    headDim,1u);

            // 6)
            if (attentionDropoutBlob && attentionDropoutBlob->GetFloat("Probability") > 0.f)
            {
                m_attentionDropout->SetErrorInput(attentionDropoutBlob, &dAtt);
                m_attentionDropout->Backwards(attentionDropoutBlob);
                Copy(&dAtt, m_attentionDropout->GetOutputError(attentionDropoutBlob).get());
            }

            // 5)
            SoftmaxBackwards(&dScores, attHeadAll, &dAtt);
            PerElementMul(&dScores, &dScores, triFiniteMask);
            Scale(&dScores, &dScores, scalarScale);

            //4b)
            MatMul_Strided<TransposeMode::None>(&dQh,
                                                &dScores,
                                                &KeyHead,
                                                dT,1u,
                                                dC,1u,
                                                headDim,1u);
            //4a)
            MatMul_Strided<TransposeMode::Left>(&dKh,
                                                &dScores,
                                                &QueryHead,
                                                dT,1u,
                                                dC,1u,
                                                headDim,1u);
        }
        // 3) Merge all head gradients back to packed QKV layout with a single op.
        MergeQKV(qkvGrad, dQhAll, dKhAll, dVhAll);

        // 2)
        Datablob<T, Mat>* attentionBlob = _blob->GetBlob("CSA_AttentionLayer");
        m_attention->SetInput(attentionBlob, lastInput);
        m_attention->SetErrorInput(attentionBlob, qkvGrad);
        m_attention->Backwards(attentionBlob);

        Copy(errorOut, m_attention->GetOutputError(attentionBlob).get());

    }

    private:
    struct AttentionDims
    {
        uint32_t C_EmbedDims = 0u;
        uint32_t T_block_size = 0u;
        uint32_t B_batch = 0u;
        uint32_t TB = 0u;
        uint32_t nHead = 0u;
        uint32_t headDim = 0u;
    };

    AttentionDims GetDims(Datablob<T, Mat>* _blob, Mat* _input) const
    {
        AttentionDims dims{};
        dims.C_EmbedDims = _input->GetDimsX();
        dims.T_block_size = _input->GetDimsY();
        dims.B_batch = _input->GetDimsZ();
        dims.TB = dims.T_block_size * dims.B_batch;
        dims.nHead = _blob->GetUInt("NumHeads");
        assert(dims.nHead > 0u);
        assert(dims.C_EmbedDims % dims.nHead == 0u);
        dims.headDim = dims.C_EmbedDims / dims.nHead;
        return dims;
    }

    std::string AttentionDropoutKey(uint32_t _head) const
    {
        return "CSA_AttentionDropout_" + std::to_string(_head);
    }

    std::string HeadAttKey(uint32_t _head) const
    {
        return "Scratch_AttHead_" + std::to_string(_head);
    }

    Datablob<T, Mat>* GetAttentionDropoutBlob(Datablob<T, Mat>* _blob, uint32_t _head) const
    {
        return _blob->GetBlob(AttentionDropoutKey(_head));
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

        const uint32_t nHead = _blob->GetUInt("NumHeads");
        for (uint32_t head = 0; head < nHead; ++head)
        {
            Datablob<T, Mat>* attentionDropoutBlob = GetAttentionDropoutBlob(_blob, head);
            _out.push_back(typename Layer<T,Mat>::sublayerinfo{AttentionDropoutKey(head), m_attentionDropout , _blob->GetBlob(AttentionDropoutKey(head))});
        }
    }

    private:
    public:
    // Member layers
    Dense<T,ActFunc,Mat, false>* m_attention        = nullptr;
    Dense<T,ActFunc,Mat, true>*  m_proj             = nullptr;
    DropoutLayer<T,Mat>*         m_attentionDropout = nullptr;
    DropoutLayer<T,Mat>*         m_residualDropout  = nullptr;
};
