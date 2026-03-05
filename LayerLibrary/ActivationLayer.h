#pragma once

#include "Layer.h"
#include "../MatrixLibrary/MatrixBase.h"
#include "../MatrixLibrary/MatrixBase_Functions.h"
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>

#include "../ToolsLibrary/Tools.h"

#include <cmath>
#include <functional>
#include <utility>

#include "Datablob.h"

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitActivationBlob(  Dims3D         _dims,
                                    std::string&   _outputName,
                                    uint32_t       _batchSize = 1,
                                    bool           _initForTraining = true)
{
    MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
    Datablob<T, Mat>* blob = new Datablob<T, Mat>();
    auto output = inst.AllocateMatrix(_dims, _outputName);
    blob->Set("Output_0", output);
    if(_initForTraining)
    {
        auto errorOut = inst.AllocateMatrix(_dims, "errorOut");
        blob->Set("ErrorOut", errorOut);
    }

    return blob;
}

template<class T, class ActFunc, class Mat = MatrixCPU<T>>
class ActivationLayer : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

    public:

    virtual std::string GetTypeName() override
    {
        return "ActivationLayer";
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

    void Forward(Datablob<T, Mat>* _blob)
    {
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        assert(input);
        assert(output);

        ActFunc act;
        act.activateMat(output, input);
    }

    void Backwards(Datablob<T, Mat>* _blob)
    {
        typename MatrixManager<T, Mat>::MatrixRef errorRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* error = errorRef.get();
        assert(error);

        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOutput_0");
        Mat* errorOut = errorOutRef.get();
        assert(output);
        assert(errorOut);

        ActFunc act;
		act.derivativeMat(errorOut, output);
        PerElementMul(errorOut, error, errorOut);
    }
};
