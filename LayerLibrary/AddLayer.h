#pragma once

#include "Layer.h"
#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/MatrixBase_Functions.h"
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>

#include "../MatrixLibrary/MatrixManager.h"

#include <cassert>
#include <cstring>

#include "Datablob.h"

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitAddBlob(bool _initForTraining = true)
{
    Datablob<T,Mat>* blob = new Datablob<T, Mat>();
    blob->Set("TrainingEnabled", _initForTraining ? 1u : 0u);
    return blob;
}

template<class T, class Mat = MatrixCPU<T>>
class AddLayer : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

public:
    AddLayer(){}

    virtual std::string GetTypeName() override
    {
        return "AddLayer";
    }
    
    uint32_t GetOutputErrorCount() override
    {
        return 2u;
    }

    uint32_t GetInputCount() override
    {
        return 2u;
    }

    MatrixRef GetOutputError(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        if (_index == 0u)
        {
            return _blob->AcquireMatrix("ErrorOutput_0");
        }
        return _blob->AcquireMatrix("ErrorOutput_1");
    }

    MatrixRef GetOutput(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        return _blob->AcquireMatrix("Output_0");
    }

    void EnsureOutputsAllocated(Datablob<T, Mat>* blob) override
    {
        Mat* left=blob->GetMatrix("Input_0");Mat* right=blob->GetMatrix("Input_1");
        if (!left || !right) return;
        this->EnsureMatrix(blob,"Output_0",left->GetShape().BroadcastWith(right->GetShape()));
        if (blob->GetUInt("TrainingEnabled")) {
            this->EnsureMatrix(blob,"ErrorOutput_0",left->GetShape());
            this->EnsureMatrix(blob,"ErrorOutput_1",right->GetShape());
        }
    }

    void Forward(Datablob<T, Mat>* blob) override
    { Add(blob->GetMatrix("Output_0"),blob->GetMatrix("Input_0"),blob->GetMatrix("Input_1")); }

    void Backwards(Datablob<T, Mat>* blob) override
    {
        Mat* error=blob->GetMatrix("ErrorInput_0");
        for (uint32_t index=0;index<2;++index) {
            Mat* output=blob->GetMatrix("ErrorOutput_"+std::to_string(index));
            const auto axes=error->GetShape().AxesReducedTo(output->GetShape());
            if (axes==MatrixAxes::None) Copy(output,error);
            else ReduceSum(output,error,axes);
        }
    }
};
