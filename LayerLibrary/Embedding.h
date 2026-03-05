#pragma once

#include "Layer.h"
#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/MatrixBase_Functions.h"
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
#include "../MatrixLibrary/MatrixManager.h"
#include "../ToolsLibrary/Tools.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>

#include "Datablob.h"

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitEmbeddingBlob(uint32_t _num_embeddings,
                               uint32_t _embed_dim,
                               uint32_t _sequence_length = 1u,
                               uint32_t _batch_size = 1u,
                               bool     _initForTraining = true,
                               bool     _random = true)
{
    Datablob<T, Mat>* blob = new Datablob<T, Mat>();
    MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

    blob->Set("Embedding_TrainingEnabled", _initForTraining ? 1u : 0u);

    auto weights = inst.AllocateMatrix({_embed_dim, _num_embeddings}, "Embedding_Weights");
    blob->Set("Embedding_Weights", weights);

    if (_initForTraining)
    {
        auto wUpdate = inst.AllocateMatrix({_embed_dim, _num_embeddings}, "Embedding_WUpdate");
        blob->Set("Embedding_WUpdate", wUpdate);
    }

    auto output = inst.AllocateMatrix({_embed_dim, _sequence_length, _batch_size}, "Embedding_Output_0");
    blob->Set("Output_0", output);
    auto scalarZero = inst.AllocateMatrix({1u, 1u, 1u}, "Embedding_ScalarZero");
    scalarZero->SetValue(0u, 0u, T(0));
    blob->Set("Embedding_ScalarZero", scalarZero);

    if (_initForTraining)
    {
        auto outputErr = inst.AllocateMatrix({1u, _sequence_length, _batch_size}, "Embedding_ErrorOut");
        blob->Set("Embedding_ErrorOut", outputErr);
    }

    if (_random)
    {
        const float stddev = 0.02f;
        auto normalGen = std::bind(RandomUtils::random_normal, 0.0f, stddev);
        MapFunction_Zero(weights.get(), normalGen);
    }
    else
    {
        Fill(weights.get(), static_cast<T>(0));
    }

    return blob;
}

template<class T, class Mat = MatrixCPU<T>>
class Embedding : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

public:

    virtual std::string GetTypeName() override
    {
        return "Embedding";
    }

    std::vector<MatrixRef>* GetWeights(Datablob<T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_weightMatrices.clear();
        if (MatrixRef weights = _blob->AcquireMatrix("Embedding_Weights"))
        {
            Layer<T, Mat>::m_weightMatrices.push_back(weights);
        }
        return &this->m_weightMatrices;
    }

    std::vector<MatrixRef>* GetGradients(Datablob< T, Mat>* _blob) override
    {
        Layer<T, Mat>::m_gradientMatrices.clear();
        if (MatrixRef wUpdate = _blob->AcquireMatrix("Embedding_WUpdate"))
        {
            Layer<T, Mat>::m_gradientMatrices.push_back(wUpdate);
        }
        return &this->m_gradientMatrices;
    }

    MatrixRef GetOutputError(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        return _blob->AcquireMatrix("Embedding_ErrorOut");
    }

    MatrixRef GetOutput(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        return _blob->AcquireMatrix("Output_0");
    }

    void EnsureOutputsAllocated(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        if (!input)
        {
            return;
        }

        typename MatrixManager<T, Mat>::MatrixRef weightsRef = _blob->AcquireMatrix("Embedding_Weights");
        Mat* weights = weightsRef.get();
        assert(weights);

        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        const uint32_t embedDim = weights->GetDimsX();
        const uint32_t tokens = input->GetDimsY();
        const uint32_t batch = input->GetDimsZ();
        const bool outputMismatch = !output
                                    || output->GetDimsX() != embedDim
                                    || output->GetDimsY() != tokens
                                    || output->GetDimsZ() != batch;
        if (outputMismatch)
        {
            if (output)
            {
                inst.RemoveMatrix(output);
            }
            auto outputRefNew = inst.AllocateMatrix({embedDim, tokens, batch}, "Embedding_Output_0");
            _blob->Set("Output_0", outputRefNew);
        }

        if (_blob->GetUInt("Embedding_TrainingEnabled") > 0u)
        {
            typename MatrixManager<T, Mat>::MatrixRef outputErrRef = _blob->AcquireMatrix("Embedding_ErrorOut");
            Mat* outputErr = outputErrRef.get();
            const bool errMismatch = !outputErr
                                     || outputErr->GetDimsX() != input->GetDimsX()
                                     || outputErr->GetDimsY() != tokens
                                     || outputErr->GetDimsZ() != batch;
            if (errMismatch)
            {
                if (outputErr)
                {
                    inst.RemoveMatrix(outputErr);
                }
                auto outputErrRefNew = inst.AllocateMatrix({input->GetDimsX(), tokens, batch}, "Embedding_ErrorOut");
                _blob->Set("Embedding_ErrorOut", outputErrRefNew);
            }
        }
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {        
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef weightsRef = _blob->AcquireMatrix("Embedding_Weights");
        Mat* weights = weightsRef.get();
        GatherRows(output, weights, input);
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef errorInRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* errorIn = errorInRef.get();
        typename MatrixManager<T, Mat>::MatrixRef errorOutRef = _blob->AcquireMatrix("Embedding_ErrorOut");
        Mat* errorOut = errorOutRef.get();
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef wUpdateRef = _blob->AcquireMatrix("Embedding_WUpdate");
        Mat* wUpdate = wUpdateRef.get();

        if (errorOut)
            Clear(errorOut);

        if (!wUpdate)
            return;

        Clear(wUpdate);
        ScatterAddRows(wUpdate, errorIn, input);
    }
};
