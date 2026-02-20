#pragma once

#include "Layer.h"
#include <MatrixLibrary/MatrixManager.h>
#include <MatrixLibrary/MatrixBase.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
#include <MatrixLibrary/CPU/MatrixCPU.h>
#include <ToolsLibrary/Tools.h>

#include <cmath>
#include <functional>
#include <utility>
#include <cstring>

#include "Datablob.h"

template<class T, class Mat = MatrixCPU<T>>
Datablob<T, Mat>* InitLayerNormBlob( bool           _initForTraining = true,
                                bool           _random          = true,
                                bool           _useBias         = true)
{
    Datablob<T, Mat>* blob           = new Datablob<T, Mat>();
    MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

    blob->Set("TrainingEnabled", _initForTraining? 1u : 0u);
    blob->Set("UseBias", _useBias ? 1u : 0u);

    return blob;
}

template<class T, class Mat = MatrixCPU<T>>
class LayerNorm : public Layer<T, Mat>
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

    public:
    std::vector<MatrixRef>* GetWeights(Datablob<T, Mat>* _blob)  override
    {
        Layer<T, Mat>::m_weightMatrices.clear();
        if (MatrixRef GScale = _blob->AcquireMatrix("GammaScale"))
        {
            Layer<T, Mat>::m_weightMatrices.push_back(GScale);
        }
        if (MatrixRef bShift = _blob->AcquireMatrix("BetaShift"))
        {
            Layer<T, Mat>::m_weightMatrices.push_back(bShift);
        }
        return &this->m_weightMatrices;
	}

    std::vector<MatrixRef>* GetGradients(Datablob< T, Mat>* _blob)  override
    {
        Layer<T, Mat>::m_gradientMatrices.clear();
        if (MatrixRef GScale_Grad = _blob->AcquireMatrix("GammaScale_Grad"))
        {
            Layer<T, Mat>::m_gradientMatrices.push_back(GScale_Grad);
        }
        if (MatrixRef bShift_Grad = _blob->AcquireMatrix("BetaShift_Grad"))
        {
            Layer<T, Mat>::m_gradientMatrices.push_back(bShift_Grad);
        }
		return &this->m_gradientMatrices;
    }

    virtual std::string GetTypeName() override
    {
        return "LayerNorm";
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

        typename MatrixManager<T, Mat>::MatrixRef gammaRef = _blob->AcquireMatrix("GammaScale");
        if (!gammaRef.get())
        {
            DynamicInit(_blob, input);
            return;
        }

        this->EnsureMatrix(_blob, "Output_0", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
        if (_blob->GetUInt("TrainingEnabled") > 0u)
        {
            this->EnsureMatrix(_blob, "XHat", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
            this->EnsureMatrix(_blob, "ErrorOut", input->GetDimsX(), input->GetDimsY(), input->GetDimsZ());
        }
    }

    void DynamicInit(Datablob<T, Mat>* _blob, Mat* _input)
    {
        // Allocate an output matrix.
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        assert(_input);

        auto gScale = inst.AllocateMatrix({_input->GetDimsX(), 1u}, "GammaScale");
        _blob->Set("GammaScale", gScale);

        // Match standard LayerNorm initialization: gamma=1, beta=0.
        Fill(gScale.get(), static_cast<T>(1));
        const bool useBias = _blob->GetUInt("UseBias") > 0u;
        if (useBias)
        {
            auto bShift = inst.AllocateMatrix({_input->GetDimsX(), 1u}, "BetaShift");
            _blob->Set("BetaShift", bShift);
            Fill(bShift.get(), static_cast<T>(0));
        }

        auto output = inst.AllocateMatrix(_input->GetDims(), "Output_0");
        _blob->Set("Output_0", output);

        if(_blob->GetUInt("TrainingEnabled") > 0u)
        {
            auto xHat = inst.AllocateMatrix(_input->GetDims(), "XHat");
            _blob->Set("XHat", xHat);

            auto outputErr = inst.AllocateMatrix(_input->GetDims(), "ErrorOut");
            _blob->Set("ErrorOut", outputErr);

            auto gScaleGrad = inst.AllocateMatrix({_input->GetDimsX(), 1u}, "GammaScale_Grad");
            _blob->Set("GammaScale_Grad", gScaleGrad);

            Fill(gScaleGrad.get(), static_cast<T>(0));
            if (useBias)
            {
                auto bShiftGrad = inst.AllocateMatrix({_input->GetDimsX(), 1u}, "BetaShift_Grad");
                _blob->Set("BetaShift_Grad", bShiftGrad);
                Fill(bShiftGrad.get(), static_cast<T>(0));
            }
        }
    }

    void Forward(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef inputRef = _blob->AcquireMatrix("Input_0");
        Mat* input = inputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef outputRef = _blob->AcquireMatrix("Output_0");
        Mat* output = outputRef.get();

        typename MatrixManager<T, Mat>::MatrixRef gScaleRef = _blob->AcquireMatrix("GammaScale");
        Mat* gScale = gScaleRef.get();
        typename MatrixManager<T, Mat>::MatrixRef bShiftRef = _blob->AcquireMatrix("BetaShift");
        Mat* bShift = bShiftRef.get();
        typename MatrixManager<T, Mat>::MatrixRef xHatRef = _blob->AcquireMatrix("XHat");
        Mat* xHat = xHatRef.get();

        // This is a standard math operation, so we just call the forward version of the function.
        const T eps = static_cast<T>(1e-5);
        LayerNormOp<T>(output, input, gScale, bShift, xHat, eps);
    }

    void Backwards(Datablob<T, Mat>* _blob) override
    {
        typename MatrixManager<T, Mat>::MatrixRef errorRef = _blob->AcquireMatrix("ErrorInput_0");
        Mat* error = errorRef.get();
        typename MatrixManager<T, Mat>::MatrixRef errorOutRef = _blob->AcquireMatrix("ErrorOut");
        Mat* errorOut = errorOutRef.get();
        typename MatrixManager<T, Mat>::MatrixRef lastInputRef = _blob->AcquireMatrix("Input_0");
        Mat* lastInput = lastInputRef.get();
        typename MatrixManager<T, Mat>::MatrixRef gScaleRef = _blob->AcquireMatrix("GammaScale");
        Mat* gScale = gScaleRef.get();
        typename MatrixManager<T, Mat>::MatrixRef gScaleGradRef = _blob->AcquireMatrix("GammaScale_Grad");
        Mat* gScaleGrad = gScaleGradRef.get();
        typename MatrixManager<T, Mat>::MatrixRef bShiftGradRef = _blob->AcquireMatrix("BetaShift_Grad");
        Mat* bShiftGrad = bShiftGradRef.get();
        typename MatrixManager<T, Mat>::MatrixRef xHatRef = _blob->AcquireMatrix("XHat");
        Mat* xHat = xHatRef.get();

        // This is a standard math operation, so we just call the backwards version of the function.
        const T eps = static_cast<T>(1e-5);
        LayerNormBackwardsOp<T>(errorOut,
                                  error,
                                  lastInput,
                                  gScale,
                                  gScaleGrad,
                                  bShiftGrad,
                                  xHat,
                                  eps);
    }

};
