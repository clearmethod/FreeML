#pragma once

#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/CPU/MatrixCPU.h"
#include "../ToolsLibrary/ThreadPool.h"
//#include "../GraphLibrary/ExecGraph.h"
#include "../ToolsLibrary/Logger.h"
#include <ToolsLibrary/Tools.h>

#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <utility>

#include "Datablob.h"

template<typename T, class Mat = MatrixCPU<T>>
class Layer
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

public:

    Layer()
    {
        m_guid = GenerateGuidUINT();
    }
    virtual ~Layer() = default;

    void BaseInit()
    {
    }

    uint64_t GetID()
    {
        return m_guid;
    }

    static_assert(std::is_base_of_v<MatrixBase<T>, Mat>,
                  "Mat must derive from MatrixBase<T>");

    // Forward and backwards passes for inference and training.
    virtual void Forward    ( Datablob<T, Mat>* _blob ) = 0;
    virtual void Backwards  ( Datablob<T, Mat>* _blob ) = 0;

    void Run(Datablob<T, Mat>* _blob)
    {
        EnsureOutputsAllocated(_blob);
        Forward(_blob);
    }

    // Get the forward result for this layer
    virtual std::string GetOutputName(Datablob<T, Mat>* _blob, uint32_t _index = 0){ return GetOutput(_blob,_index) ? GetOutput(_blob,_index)->GetName() : "";}

    virtual MatrixRef GetOutput(Datablob<T, Mat>* _blob, uint32_t _index = 0) = 0;
    // Get the backward pass calculated output error for this layer
    virtual MatrixRef GetOutputError(Datablob<T, Mat>* _blob, uint32_t _index = 0) = 0;
    // Gets a vector of all the trainable parameters in this layer.
    virtual std::vector<MatrixRef>* GetWeights(Datablob<T, Mat>* _blob)    { return &m_weightMatrices; };
    // Gets the gradients caclulated in the last backwards pass for this layer.
    virtual std::vector<MatrixRef>* GetGradients(Datablob<T, Mat>* _blob)  { return &m_gradientMatrices; };

    virtual uint32_t GetOutputErrorCount() {return 1u;};
    virtual uint32_t GetInputCount() { return 1u;};

    void SetName(std::string _name)
    {
        m_name = _name;
    }
    std::string GetName() const
    {
        return m_name;
	}

    virtual std::string Description(Datablob<T, Mat>* _blob)
    {
        std::stringstream ss;
        ss << m_name << ": No Description() defined for this layer";
        return ss.str();
    }

    virtual std::string GetString(Datablob<T, Mat>* _blob)
    {
        return "No String Set";
    }

    virtual std::string GetInputOutputString(Datablob<T, Mat>* _blob)
    {
        auto* inMat = _blob->GetMatrix(GetInputName());
        auto* outMat = _blob->GetMatrix(GetOutputName());
        std::string in = inMat ? inMat->GetDims().GetString() : "TBD";
        std::string out = outMat ? outMat->GetDims().GetString() : "TBD";
        return GetName() + ": " + in + " -> " + out;
    }

    virtual void Print(Datablob<T, Mat>* _blob)
    {
        LOG_INFO() << Description(_blob);
    } 

    virtual std::string GetTypeName() = 0;

    struct sublayerinfo
    {
        std::string      name;
        Layer<T,Mat>*    layer;
        Datablob<T,Mat>* data;
    };

    virtual std::string GetInputName(uint32_t _index = 0)
    {
        return "Input_" + std::to_string(_index);
    }

    virtual std::string GetOutputName(uint32_t _index = 0)
    {
        return "Output_" + std::to_string(_index);
    }

    virtual std::string GetErrorOutputName(uint32_t _index = 0)
    {
        return "ErrorOutput_" + std::to_string(_index);
    }

    virtual std::string GetErrorInputName(uint32_t _index = 0)
    {
        return "ErrorInput_" + std::to_string(_index);
    }

    virtual void SetInput(Datablob<T, Mat>* _blob, MatrixRef& _mat, uint32_t _index = 0)
    {
        std::string inputName = GetInputName(_index);
        _blob->Set(inputName, _mat);
    }

    virtual void SetInput(Datablob<T, Mat>* _blob, Mat* _mat, uint32_t _index = 0)
    {
        if (!_mat)
        {
            return;
        }
        MatrixRef ref = MatrixManager<T, Mat>::Instance().Acquire(_mat);
        if (!ref)
        {
            return;
        }
        SetInput(_blob, ref, _index);
    }

    virtual void SetErrorInput(Datablob<T, Mat>* _blob, const MatrixRef& _mat, uint32_t _index = 0)
    {
        std::string inputName = GetErrorInputName(_index);
        _blob->Set(inputName, _mat);
    }
    virtual void SetErrorInput(Datablob<T, Mat>* _blob, Mat* _mat, uint32_t _index = 0)
    {
        std::string inputName = GetErrorInputName(_index);
        _blob->Set(inputName, _mat);
    }

    //virtual void SetErrorInput(Datablob<T, Mat>* _blob, Mat* _mat, uint32_t _index = 0)
    //{
    //    if (!_mat)
    //    {
    //        return;
    //    }
    //    MatrixRef ref = MatrixManager<T, Mat>::Instance().Acquire(_mat);
    //    if (!ref)
    //    {
    //        return;
    //    }
    //    SetErrorInput(_blob, ref, _index);
    //}

    virtual void EnsureOutputsAllocated(Datablob<T, Mat>* _blob)
    {
    }

    void EnsureMatrix(Datablob<T, Mat>* _blob, const char* _key, uint32_t _dimsX, uint32_t _dimsY = 1, uint32_t _dimsZ = 1)
    {
        auto matRef = _blob->AcquireMatrix(_key);
        Mat* mat = matRef.get();
        if (!mat || mat->GetDimsX() != _dimsX || mat->GetDimsY() != _dimsY || mat->GetDimsZ() != _dimsZ)
        {
            MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
            if (mat)
            {
                inst.RemoveMatrix(mat);
            }
            auto newRef = inst.AllocateMatrix({_dimsX, _dimsY, _dimsZ}, _key);
            _blob->Set(_key, newRef);
        }
    }

    virtual void GetSublayerPairs(std::vector<sublayerinfo>& _out,
                          Datablob<T,Mat>* _blob)
    {
        
    }

    virtual std::string GetMetaData()
    {
        return "";
    }

    std::string m_name;
    uint64_t    m_guid;

    // Output Vectors
    std::vector<MatrixRef> m_weightMatrices;
    std::vector<MatrixRef> m_gradientMatrices;
};

template<typename Derived, typename T, class Mat = MatrixCPU<T>>
class LayerInit : public Layer<T, Mat>
{
public:
    template<typename... Args>
    void Init(Args&&... args)
    {
        static_cast<Derived*>(this)->_Init(std::forward<Args>(args)...);
        Layer<T, Mat>::BaseInit();
    }
};
