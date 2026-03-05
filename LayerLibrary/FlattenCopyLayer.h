#pragma once

#include "Layer.h"
#include "Datablob.h"

#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/MatrixManager.h>
#include <MatrixLibrary/MatrixBase.h>
#include <MatrixLibrary/CPU/MatrixCPU.h>

#include <cassert>
#include <string>

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitFlattenCopyBlob(bool _initForTraining = true)
{
    Datablob<T, Mat>* blob = new Datablob<T, Mat>();
    blob->Set("TrainingEnabled", _initForTraining ? 1u : 0u);
    return blob;
}

template<class T, class Mat = MatrixCPU<T>>
class FlattenCopyLayer : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

    public:
    std::string GetTypeName() override { return "FlattenCopy"; }

    uint32_t GetOutputErrorCount() override { return 1u; }

    MatrixRef GetOutput(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        return _blob->AcquireMatrix("Output_0");
    }

    MatrixRef GetOutputError(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        return _blob->AcquireMatrix("ErrorOutput_0");
    }

    void EnsureOutputsAllocated(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        if (!input)
        {
            return;
        }

        const uint32_t dimsX = input->GetDimsX();
        const uint32_t dimsY = input->GetDimsY() * input->GetDimsZ();

        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        typename MatrixManager<T, Mat>::MatrixRef outRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outRef.get();
        const bool outputMismatch = !output
                                    || output->GetDimsX() != dimsX
                                    || output->GetDimsY() != dimsY
                                    || output->GetDimsZ() != 1u;
        if (outputMismatch)
        {
            if (output)
            {
                inst.RemoveMatrix(output);
            }
            auto outputRefNew = inst.AllocateMatrix({dimsX, dimsY}, "Flatten_Output");
            _blob->Set("Output_0", outputRefNew);
        }

        if (_blob->GetUInt("TrainingEnabled") > 0u)
        {
            typename MatrixManager<T, Mat>::MatrixRef errRef = _blob->AcquireMatrix("ErrorOutput_0");
            Mat* errorOut = errRef.get();
            const bool errMismatch = !errorOut
                                     || errorOut->GetDimsX() != input->GetDimsX()
                                     || errorOut->GetDimsY() != input->GetDimsY()
                                     || errorOut->GetDimsZ() != input->GetDimsZ();
            if (errMismatch)
            {
                if (errorOut)
                {
                    inst.RemoveMatrix(errorOut);
                }
                auto errorRefNew = inst.AllocateMatrix({input->GetDimsX(), input->GetDimsY(), input->GetDimsZ()},
                    "Flatten_ErrorOut");
                _blob->Set("ErrorOutput_0", errorRefNew);
            }
        }
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();

        Copy(output, input);
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef errorRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* error = errorRef.get();
        typename MatrixManager<T, Mat>::MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOutput_0");
        Mat* errorOut = errorOutRef.get();

        Copy(errorOut, error);
    }
};
