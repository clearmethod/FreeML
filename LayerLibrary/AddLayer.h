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
            return _blob->AcquireMatrix("ErrorOut_0");
        }
        return _blob->AcquireMatrix("ErrorOut_1");
    }

    MatrixRef GetOutput(Datablob<T, Mat>* _blob, uint32_t _index = 0) override
    {
        return _blob->AcquireMatrix("Output_0");
    }

    void EnsureOutputsAllocated(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef input0Ref = _blob->AcquireMatrix("Input_0");
        Mat* input0 = input0Ref.get();
        typename MatrixManager<T, Mat>::MatrixRef input1Ref = _blob->AcquireMatrix("Input_1");
        Mat* input1 = input1Ref.get();
        if (!input0 || !input1)
        {
            return;
        }

        assert(input0->GetDimsX() == input1->GetDimsX());
        assert(input0->GetDimsY() == input1->GetDimsY());
        const uint32_t outZ = std::max(input0->GetDimsZ(), input1->GetDimsZ());
        assert(input0->GetDimsZ() == outZ || input0->GetDimsZ() == 1u);
        assert(input1->GetDimsZ() == outZ || input1->GetDimsZ() == 1u);

        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        const bool outputMismatch = !output
                                    || output->GetDimsX() != input0->GetDimsX()
                                    || output->GetDimsY() != input0->GetDimsY()
                                    || output->GetDimsZ() != outZ;
        if (outputMismatch)
        {
            if (output)
            {
                inst.RemoveMatrix(output);
            }
            auto outputRefNew = inst.AllocateMatrix({input0->GetDimsX(), input0->GetDimsY(), outZ}, "Output_0");
            _blob->Set("Output_0", outputRefNew);
        }

        if (_blob->GetUInt("TrainingEnabled") > 0u)
        {
            typename MatrixManager<T, Mat>::MatrixRef err0Ref = _blob->AcquireMatrix("ErrorOut_0");
            Mat* err0 = err0Ref.get();
            const bool err0Mismatch = !err0
                                      || err0->GetDimsX() != input0->GetDimsX()
                                      || err0->GetDimsY() != input0->GetDimsY()
                                      || err0->GetDimsZ() != input0->GetDimsZ();
            if (err0Mismatch)
            {
                if (err0)
                {
                    inst.RemoveMatrix(err0);
                }
                auto err0RefNew = inst.AllocateMatrix(input0->GetDims(), "ErrorOut_0");
                _blob->Set("ErrorOut_0", err0RefNew);
            }

            typename MatrixManager<T, Mat>::MatrixRef err1Ref = _blob->AcquireMatrix("ErrorOut_1");
            Mat* err1 = err1Ref.get();
            const bool err1Mismatch = !err1
                                      || err1->GetDimsX() != input1->GetDimsX()
                                      || err1->GetDimsY() != input1->GetDimsY()
                                      || err1->GetDimsZ() != input1->GetDimsZ();
            if (err1Mismatch)
            {
                if (err1)
                {
                    inst.RemoveMatrix(err1);
                }
                auto err1RefNew = inst.AllocateMatrix(input1->GetDims(), "ErrorOut_1");
                _blob->Set("ErrorOut_1", err1RefNew);
            }
        }
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef input0Ref = _blob->AcquireMatrix("Input_0");
        Mat* input0 = input0Ref.get();
        typename MatrixManager<T, Mat>::MatrixRef input1Ref = _blob->AcquireMatrix("Input_1");
        Mat* input1 = input1Ref.get();
        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();
        assert(input0);
        assert(input1);
        assert(output);

        const uint32_t outZ = output->GetDimsZ();
        const uint32_t in0Z = input0->GetDimsZ();
        const uint32_t in1Z = input1->GetDimsZ();

        if (in0Z == outZ && in1Z == outZ)
        {
            Add(output, input0, input1);
            return;
        }

        for (uint32_t z = 0u; z < outZ; ++z)
        {
            Mat outSlice;
            output->GetSliceZ(&outSlice, z);

            Mat lhsSlice;
            input0->GetSliceZ(&lhsSlice, in0Z == 1u ? 0u : z);

            Mat rhsSlice;
            input1->GetSliceZ(&rhsSlice, in1Z == 1u ? 0u : z);

            Add(&outSlice, &lhsSlice, &rhsSlice);
        }
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef errorInRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* errorIn = errorInRef.get();
        typename MatrixManager<T, Mat>::MatrixRef errorOut0Ref = _blob->AcquireMatrix("ErrorOut_0");
        Mat* errorOut0 = errorOut0Ref.get();
        typename MatrixManager<T, Mat>::MatrixRef errorOut1Ref = _blob->AcquireMatrix("ErrorOut_1");
        Mat* errorOut1 = errorOut1Ref.get();
        typename MatrixManager<T, Mat>::MatrixRef input0Ref = _blob->AcquireMatrix("Input_0");
        Mat* input0 = input0Ref.get();
        typename MatrixManager<T, Mat>::MatrixRef input1Ref = _blob->AcquireMatrix("Input_1");
        Mat* input1 = input1Ref.get();
        assert(errorIn);
        assert(errorOut0);
        assert(errorOut1);
        assert(input0);
        assert(input1);

        const uint32_t dX = errorIn->GetDimsX();
        const uint32_t dY = errorIn->GetDimsY();
        const uint32_t outZ = errorIn->GetDimsZ();
        const uint32_t in0Z = input0->GetDimsZ();
        const uint32_t in1Z = input1->GetDimsZ();
        const uint32_t slice = dX * dY;


        if (in0Z == outZ)
        {
            Copy(errorOut0, errorIn);
        }
        else
        {
            const T* errPtr = errorIn->DataRead();
            T* err0Ptr = errorOut0->DataWrite();
            T* err1Ptr = errorOut1->DataWrite();

            std::memset(err0Ptr, 0, sizeof(T) * errorOut0->GetElementCount());
            for (uint32_t z = 0; z < outZ; ++z)
            {
                const T* src = errPtr + z * slice;
                for (uint32_t i = 0; i < slice; ++i)
                {
                    err0Ptr[i] += src[i];
                }
            }
        }

        if (in1Z == outZ)
        {
            Copy(errorOut1, errorIn);
        }
        else
        {
            const T* errPtr = errorIn->DataRead();
            T* err0Ptr = errorOut0->DataWrite();
            T* err1Ptr = errorOut1->DataWrite();

            std::memset(err1Ptr, 0, sizeof(T) * errorOut1->GetElementCount());
            for (uint32_t z = 0; z < outZ; ++z)
            {
                const T* src = errPtr + z * slice;
                for (uint32_t i = 0; i < slice; ++i)
                {
                    err1Ptr[i] += src[i];
                }
            }
        }
    }
};
