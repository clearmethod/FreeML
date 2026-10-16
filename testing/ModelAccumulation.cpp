#include "LayerTestSupport.h"
#include <ModelLibrary/Model.h>

using namespace LayerTesting;
using M = TestMatrix;

// Capture the gradient presented to the optimizer, then apply a real update.
class RecordingOptimiser : public Optimiser<float, M>
{
public:
    std::vector<Values> gradients;
    void Step(M* parameter, M* gradient) override
    {
        gradients.push_back(Read(gradient));
        auto values = Read(parameter);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] += 0.001f * gradients.back()[i];
        Write(parameter, values);
    }
};

int main()
{
    try
    {
        LOG_INFO() << "Starting gradient accumulation regression";
#if defined(LAYER_TEST_Metal)
        if (!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if (!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        auto& manager = MatrixManager<float, M>::Instance();
        auto input = manager.AllocateMatrix(MatrixShape::Matrix(1,1), "AccumulationInput");
        auto target = manager.AllocateMatrix(MatrixShape::Matrix(1,1), "AccumulationTarget");
        Write(target.get(), {0.0f});

        Model<float, M> model;
        auto* first = new Dense<float, Identity<float>, M>;
        auto* second = new Dense<float, Identity<float>, M>;
        first->SetName("First");
        second->SetName("Second");
        auto* firstBlob = InitDenseBlob<float, M>(1, 1, 1, true, false, false);
        auto* secondBlob = InitDenseBlob<float, M>(1, 1, 1, true, false, false);
        auto weight = firstBlob->AcquireMatrix("Dense_Weights");
        auto oldWeight = secondBlob->AcquireMatrix("Dense_Weights");
        secondBlob->Set("Dense_Weights", weight);
        manager.RemoveMatrix(oldWeight);
        Write(weight.get(), {2.0f});
        model.AddLayer(first, firstBlob);
        model.AddLayer(second, secondBlob);
        model.AddDependency(second, 0, first, 0);
        model.SetExpectedInputDims(MatrixShape::Matrix(1,1));
        model.SetFinalOutputLayer(second);
        model.Init();
        auto* optimizer = new RecordingOptimiser;
        model.SetOptimiser(optimizer);

        auto run = [&](float value, bool training = true)
        {
            Write(input.get(), {value});
            model.Run(input.get(), training ? target.get() : nullptr);
        };
        // Output = w*w*x. DirectLoss gradient = -output for target zero.
        // Each tied occurrence contributes -w^3*x^2, so the total is -2*w^3*x^2.
        auto expected = [](float w, float meanSquaredInput)
        {
            return -2.0f * w * w * w * meanSquaredInput;
        };
        auto checkUpdate = [&](size_t count, float gradient)
        {
            Require(optimizer->gradients.size() == count, "Unexpected number of optimizer updates");
            Expect(optimizer->gradients.back(), {gradient}, "Averaged tied gradient");
        };

        Require(model.GetGradientAccumulationSteps() == 1, "Default must update every pass");
        model.SetGradientAccumulationSteps(3);
        run(1.0f);
        run(2.0f);
        Require(optimizer->gradients.empty(), "Optimizer must wait for a complete window");
        Expect(Read(weight.get()), {2.0f}, "Parameters must stay unchanged before an update");
        run(9.0f, false);
        Require(model.GetPendingGradientPasses() == 2, "Inference must not contribute a sample");
        run(3.0f);
        checkUpdate(1, expected(2.0f, 14.0f / 3.0f));
        Require(model.GetPendingGradientPasses() == 0, "Complete update must reset the counter");

        float w = Read(weight.get())[0];
        run(2.0f);
        run(4.0f);
        model.FlushAccumulatedGradients();
        checkUpdate(2, expected(w, 10.0f));
        model.FlushAccumulatedGradients();
        Require(optimizer->gradients.size() == 2, "An empty flush must be a no-op");

        // Reusing the buffers must not carry gradients over from the previous window.
        w = Read(weight.get())[0];
        run(1.0f);
        run(1.0f);
        run(1.0f);
        checkUpdate(3, expected(w, 1.0f));

        // Configuration changes finish the old window before using the new size.
        w = Read(weight.get())[0];
        run(2.0f);
        model.SetGradientAccumulationSteps(1);
        checkUpdate(4, expected(w, 4.0f));
        w = Read(weight.get())[0];
        run(3.0f);
        checkUpdate(5, expected(w, 9.0f));

        model.SetGradientAccumulationSteps(3);
        run(5.0f);
        model.ResetGradientAccumulation();
        model.FlushAccumulatedGradients();
        Require(optimizer->gradients.size() == 5, "Reset must discard pending samples");
        w = Read(weight.get())[0];
        run(1.0f);
        model.FlushAccumulatedGradients();
        checkUpdate(6, expected(w, 1.0f));
        bool rejected = false;
        try { model.SetGradientAccumulationSteps(0); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "Zero accumulation steps must be rejected");
        std::cout << "PASS: gradient accumulation, tied parameters, inference and partial windows\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
