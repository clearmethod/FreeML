#include <MatrixLibrary/CPU/MatrixCPU.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/MatrixManager.h>

#include <LayerLibrary/Dense.h>
#include <ActivationLibrary/Activations.h>
#include <ModelLibrary/Model.h>

#include <OptimiserLibrary/Optimisers.h>
#include <LossLibrary/Losses.h>

#include <ToolsLibrary/Logger.h>
#include <ToolsLibrary/ThreadPool.h>

#define T float
using MatType = MatrixCPU<T>;
using MatrixRef = typename MatrixManager<T, MatType>::MatrixRef;
using ActFunc = Tanh<T>;

int main()
{
    ThreadPool pool;

    Model<T, MatType> m;
    m.SetName("HelloFreeML");

    auto* denseData0 = InitDenseBlob<T, MatType>(4, 4, 1);
    auto* denseLayer0 = new Dense<T, ActFunc, MatType>();
    denseLayer0->SetName("Dense0");
    m.AddLayer(denseLayer0, denseData0);

    auto* denseData1 = InitDenseBlob<T, MatType>(4, 4, 1);
    auto* denseLayer1 = new Dense<T, Identity<T>, MatType>();
    denseLayer1->SetName("Dense1");
    m.AddLayer(denseLayer1, denseData1);

    m.AddDependency(denseLayer1, 0, denseLayer0, 0);
    m.SetFinalOutputLayer(denseLayer1);

    auto* optimiser = new AdamOptimiser<T, MatType>(0.1f);
    m.SetOptimiser(optimiser);

    auto* lossfunc = new MeanSquaredError<T, MatType>();
    m.SetLoss(lossfunc);

    m.Init();

    auto& inst = MatrixManager<T, MatType>::Instance();
    MatrixRef input = inst.AllocateMatrix({4, 1}, "Input");
    MatrixRef target = inst.AllocateMatrix({4, 1}, "Output");

    Fill(input.get(), 1.0f);
    Fill(target.get(), 4.0f);

    for(uint32_t i = 0; i < 100; i++)
    {
        m.Run(input.get(), target.get());
        if(i % 5 == 0)
            LOG_INFO() << "Loss: " << m.GetLastLoss();
    }
    LOG_INFO() << "Target: " << target->GetString();
    LOG_INFO() << "Output: " << m.GetOutput()->GetString();
    
    LOG_INFO() << m.GetString();
    LOG_INFO() << m.GetModelInOut();
    LOG_INFO() << inst.GetString();

    return 0;
}
