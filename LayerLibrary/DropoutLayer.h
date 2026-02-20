#pragma once

#include "Layer.h"
#include <MatrixLibrary/MatrixManager.h>
#include <MatrixLibrary/MatrixBase.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
#include <MatrixLibrary/CPU/MatrixCPU.h>

#include <cmath>
#include <functional>
#include <utility>

#include <ToolsLibrary/Tools.h>

#include "Datablob.h"

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitDropoutBlob(float _probability = 0.01f, bool _initForTraining = true )
{
    assert(_probability != 1.0f);
    Datablob<T, Mat>* blob = new Datablob<T, Mat>();

    MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
    blob->Set("TrainingEnabled", _initForTraining? 1u : 0u);
    blob->Set("Probability", _probability);

    auto scalarScale = inst.AllocateMatrix({1u, 1u, 1u}, "ScalarScale");
    scalarScale->SetValue(0u, 0u, T(1) / (T(1) - _probability));
    blob->Set("ScalarScale", scalarScale);

    auto scalarZero = inst.AllocateMatrix({1u, 1u, 1u}, "ScalarZero");
    auto scalarOne = inst.AllocateMatrix({1u, 1u, 1u}, "ScalarOne");
    scalarZero->SetValue(0u, 0u, T(0));
    scalarOne->SetValue(0u, 0u, T(1));
    blob->Set("ScalarZero", scalarZero);
    blob->Set("ScalarOne", scalarOne);
    return blob;
}

template<class T, class Mat = MatrixCPU<T>>
class DropoutLayer : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

    public:
    virtual std::string GetTypeName() override
    {
        return "DropoutLayer";
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
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        if (!input)
        {
            return;
        }

        this->EnsureMatrix(_blob, "Output_0", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
        if (_blob->GetUInt("TrainingEnabled") > 0u)
        {
            this->EnsureMatrix(_blob, "Mask", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
            this->EnsureMatrix(_blob, "ErrorOut", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
        }
    }

    void DynamicInit(Datablob<T, Mat>* _blob, Mat* _input)
    {
        // Allocate an output matrix.
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

        auto output = inst.AllocateMatrix(_input->GetDims(), "Output_0");
        _blob->Set("Output_0", output);

        if(_blob->GetUInt("TrainingEnabled") > 0u)
        {
            auto mask = inst.AllocateMatrix(_input->GetDims(), "Mask");
            _blob->Set("Mask", mask);

            auto outputErr = inst.AllocateMatrix(_input->GetDims(), "ErrorOut");
            _blob->Set("ErrorOut", outputErr);
        }
    }

    void UpdateMask(Mat* _mask, float _prob, Mat* _zero, Mat* _one)
    {
        assert(_mask);
        assert(_zero);
        assert(_one);
        const uint32_t elemCount = _mask->GetElementCount();
        if (elemCount == 0) return;

        if (_prob <= 0.0f)
        {
            Fill(_mask, _one);
            return;
        }

        if (_prob >= 1.0f)
        {
            Fill(_mask, _zero);
            return;
        }

        T* outPtr = _mask->DataWrite();
        for (uint32_t i = 0; i < elemCount; ++i)
        {
            const float r = RandomUtils::random_uniform_buffered();
            outPtr[i] = (r < _prob) ? T(0) : T(1);
        }
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();

        uint32_t isTraining = _blob->GetUInt("TrainingEnabled");
        if(isTraining)
        {
            typename MatrixManager<T, Mat>::MatrixRef maskRef = _blob->AcquireMatrix("Mask");
            Mat* mask = maskRef.get();
            typename MatrixManager<T, Mat>::MatrixRef scaleRef = _blob->AcquireMatrix("ScalarScale");
            Mat* scale = scaleRef.get();
            typename MatrixManager<T, Mat>::MatrixRef zeroRef = _blob->AcquireMatrix("ScalarZero");
            Mat* zero = zeroRef.get();
            typename MatrixManager<T, Mat>::MatrixRef oneRef = _blob->AcquireMatrix("ScalarOne");
            Mat* one = oneRef.get();
            float probs  = _blob->GetFloat("Probability");
            UpdateMask(mask, probs, zero, one);
            PerElementMul(output, input, mask);
            Scale(output, output, scale);
        }
        else
            Copy(output, input);
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef errorInRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* errorIn = errorInRef.get();
        typename MatrixManager<T, Mat>::MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOut");
        Mat* errorOut = errorOutRef.get();
        typename MatrixManager<T, Mat>::MatrixRef maskRef = _blob->AcquireMatrix("Mask");
        Mat* mask = maskRef.get();
        typename MatrixManager<T, Mat>::MatrixRef scaleRef = _blob->AcquireMatrix("ScalarScale");
        Mat* scale = scaleRef.get();

        float probs = _blob->GetFloat("Probability");

        PerElementMul(errorOut, errorIn, mask);
        Scale(errorOut, errorOut, scale);
    }

};
