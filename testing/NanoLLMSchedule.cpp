#include "LayerTestSupport.h"
#include <Examples/NanoLLM/LearningRateSchedule.h>
#include <Loading/SaveLoader.h>
#include <chrono>

using namespace LayerTesting;
using M = TestMatrix;

// SaveLoader writes relative to the working directory. Use an isolated fixture
// directory so this test never touches a user checkpoint.
struct CheckpointDirectory
{
    std::filesystem::path original = std::filesystem::current_path();
    std::filesystem::path temporary = original / ("NanoLLMScheduleTest-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    CheckpointDirectory()
    {
        if (!std::filesystem::create_directory(temporary))
            throw std::runtime_error("Cannot create checkpoint fixture directory");
        std::filesystem::current_path(temporary);
    }
    ~CheckpointDirectory()
    {
        std::filesystem::current_path(original);
        std::filesystem::remove_all(temporary);
    }
};

int main()
{
    try {
        LOG_INFO() << "Starting NanoLLM learning-rate schedule tests";
#if defined(LAYER_TEST_Metal)
        if (!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if (!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        const NanoLLMLearningRateSchedule schedule{0.001f,0.0001f,100,5000};
        if (!schedule.IsValid()) throw std::runtime_error("Valid schedule rejected");
        auto near = [](float actual, double expected) {
            if (std::abs(actual-expected) > 1e-9)
                throw std::runtime_error("Incorrect scheduled learning rate");
        };
        near(schedule.AtUpdate(0),0.001/101);
        near(schedule.AtUpdate(99),0.001*100/101);
        near(schedule.AtUpdate(100),0.001);
        near(schedule.AtUpdate(2550),0.00055);
        near(schedule.AtUpdate(5000),0.0001);
        near(schedule.AtUpdate(50000),0.0001);
        for (uint64_t i=1;i<=100;++i)
            if (schedule.AtUpdate(i) < schedule.AtUpdate(i-1))
                throw std::runtime_error("Warmup decreased");
        for (uint64_t i=101;i<=5000;++i)
            if (schedule.AtUpdate(i) > schedule.AtUpdate(i-1))
                throw std::runtime_error("Cosine decay increased");
        near(NanoLLMLearningRateSchedule{0.001f,0.0001f,0,100}.AtUpdate(0),0.001);
        for (auto invalid : {NanoLLMLearningRateSchedule{0.001f,0.002f,100,5000},
                             NanoLLMLearningRateSchedule{0.001f,0,100,5000},
                             NanoLLMLearningRateSchedule{0.001f,0.0001f,100,100}})
            if (invalid.IsValid()) throw std::runtime_error("Invalid schedule accepted");

        auto& manager = MatrixManager<float,M>::Instance();
        auto parameter = manager.AllocateMatrix(MatrixShape::Matrix(1,1),"ScheduledParameter");
        auto gradient = manager.AllocateMatrix(MatrixShape::Matrix(1,1),"ScheduledGradient");
        Write(parameter.get(),{1.0f}); Write(gradient.get(),{0.25f});
        AdamWOptimiser<float,M> adam;
        if (adam.GetMaximumStepCount()!=0) throw std::runtime_error("Fresh optimizer has steps");
        for (uint64_t update=0;update<120;++update) {
            adam.SetLearningRate(schedule.AtUpdate(update));
            adam.Step(parameter.get(),gradient.get());
        }
        CheckpointDirectory directory;
        SaveLoader<float,M> loader;
        const std::vector<typename MatrixManager<float,M>::MatrixRef> parameters{parameter};
        const auto filename = loader.SaveOptimiser(&adam,parameters);
        std::unique_ptr<Optimiser<float,M>> restored(loader.LoadOptimiser(filename,parameters));
        auto* resumed = dynamic_cast<AdamWOptimiser<float,M>*>(restored.get());
        if (!resumed || resumed->GetMaximumStepCount()!=120)
            throw std::runtime_error("Checkpoint lost schedule progress");
        resumed->SetLearningRate(schedule.AtUpdate(resumed->GetMaximumStepCount()));
        near(resumed->GetLearningRate(),schedule.AtUpdate(120));
        resumed->Step(parameter.get(),gradient.get());
        if (resumed->GetMaximumStepCount()!=121)
            throw std::runtime_error("Resumed optimizer did not advance");
        std::cout << "PASS: schedule boundaries and checkpoint continuation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
