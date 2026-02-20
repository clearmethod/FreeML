
// Matrices
#include <MatrixLibrary/CPU/MatrixCPU.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/MatrixManager.h>

#include <LayerLibrary/Dense.h>
#include <LayerLibrary/Conv2D.h>
#include <LayerLibrary/ActivationLayer.h>

#include <ActivationLibrary/Activations.h>
#include <ModelLibrary/Model.h>

#include <OptimiserLibrary/AdamOptimiser.h>
#include <LossLibrary/Losses.h>

#include <MatrixLibrary/GPU/DirectX11/DirectX11Manager.h>
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11.h>

#define T float
#ifdef _WIN32
using MatType = MatrixDX11<T>;
#else
using MatType = MatrixCPU<PRECISION>;
#endif

int main()
{
    ThreadPool pool;
    MatrixManager<T, MatType>& inst = MatrixManager<T, MatType>::Instance();
    using MatrixRef = typename MatrixManager<T, MatType>::MatrixRef;
    const uint32_t batchSize = 4;

    // Build Training Data
    ////////////////////////
    std::vector<MatrixRef> inputsUnbatched;

    // 2xBatchsize matrix.
    // We set all 4 combinations of inputs
    auto newInput0 = inst.AllocateMatrix({2, 1}, "Input");
    newInput0->SetValue(0, 0, 0.0); // 0, 0
    newInput0->SetValue(1, 0, 0.0);
    inputsUnbatched.push_back(newInput0);

    auto newInput1 = inst.AllocateMatrix({2, 1}, "Input");
    newInput1->SetValue(0, 0, 1.0); // 1, 0
    newInput1->SetValue(1, 0, 0.0);
    inputsUnbatched.push_back(newInput1);

    auto newInput2 = inst.AllocateMatrix({2, 1}, "Input");
    newInput2->SetValue(0, 0, 0.0); // 0, 1
    newInput2->SetValue(1, 0, 1.0);
    inputsUnbatched.push_back(newInput2);

    auto newInput3 = inst.AllocateMatrix({2, 1}, "Input");
    newInput3->SetValue(0, 0, 1.0); // 1, 1
    newInput3->SetValue(1, 0, 1.0);
    // Add to the array of inputs (Note: we have only one input in this demo)
    inputsUnbatched.push_back(newInput3);

    std::vector<MatrixRef> inputs;

    for(uint32_t i =0; i <= inputsUnbatched.size() - batchSize; i+=batchSize)
    {
        auto batch = inst.AllocateMatrix({2, batchSize}, "InputBatch");
        std::vector<MatType*> mats;
        for(uint32_t j = 0; j < batchSize; j++)
        {
            mats.push_back(inputsUnbatched[i + j].get());
        }
        VConcat(batch.get(), mats);
        inputs.push_back(batch);
    }

    // Use the input to build the outputs.
    std::vector<MatrixRef> outputsUnbatched;
    uint32_t counter = 0u;
    for (auto& v : inputsUnbatched)
    {
        auto newOutput = inst.AllocateMatrix({2, 1}, "TargetOutput_" + std::to_string(counter));
        counter++;
        Fill(newOutput.get(), 0.0f);
        if (v->GetValue(0, 0) == 1.0 || v->GetValue(1, 0) == 1.0)
        {
            newOutput->SetValue(0, 0,   1.0);
            newOutput->SetValue(1, 0,  -1.0);
        }
        else
        {
            newOutput->SetValue(0, 0,  -1.0);
            newOutput->SetValue(1, 0,   1.0);
        }
        outputsUnbatched.push_back(newOutput);
    }

    std::vector<MatrixRef> outputs;
    for(uint32_t i =0; i <= outputsUnbatched.size() - batchSize; i+=batchSize)
    {
        auto batch = inst.AllocateMatrix({2, batchSize}, "InputBatch");
        std::vector<MatType*> mats;
        for(uint32_t j = 0; j < batchSize; j++)
        {
            mats.push_back(outputsUnbatched[i + j].get());
        }
        VConcat(batch.get(), mats);
        outputs.push_back(batch);
    }

    // Setup Optimiser
    //////////////////
    AdamOptimiser<T, MatType> adamopt(1.0);

    // Create Layers and Model
    //////////////////////////
    // One Dense layer with two inputs and two outputs.
    const uint32_t outputsize = outputs[0].get()->GetDimsX();
    using actFunction = Tanh<T>;

    // Dense layer with no bias.
    Datablob<T, MatType>* denseData0 = InitDenseBlob<T, MatType>( outputsize, 2, batchSize,true, true, true);
    auto*                layer0     = new Dense<float, actFunction, MatType>();
    layer0->SetName("layer0");

    // Assemble the model
    Model<T, MatType> m;
    m.SetName("ORGateModel");
    m.AddLayer(layer0, denseData0);
    m.SetFinalOutputLayer(layer0);
    m.SetOptimiser(&adamopt);
    m.SetExpectedInputDims(inputs[0].get()->GetDims());
    m.Init();

    // Log Starting State
    LOG_INFO() << m.GetString();
    LOG_INFO() << inst.GetString();

    #ifdef _WIN32
        LOG_INFO() << DirectX11Manager::Instance()->GetString();
        LOG_INFO() << DirectX11Manager::Instance()->GetMemoryString();
    #endif

    // Run Training
    int count = 0;
    while (count < 10000)
    {
        for(uint32_t i = 0; i < inputs.size(); i++)
        {
            //LOG_INFO() << inputs[i]->GetString();
            m.Run(inputs[i].get(), outputs[i].get());
            if(count % 1 == 0)
            {
                //LOG_INFO() << "Input" << i << ": ";
                //LOG_INFO() << inputs[i].get()->GetString();
                //LOG_INFO() << m.GetOutput()->GetString();
                //LOG_INFO() << outputs[i].get()->GetString();
                LOG_INFO() << "Loss: " << m.GetLastLoss();
                //LOG_INFO() << denseData0->GetMatrix<MatType>("Weights")->GetString();
                //LOG_INFO() << denseData0->GetMatrix<MatType>("Bias")->GetString();
                
            }
            count++;
        }
    }

    return 0;
}
