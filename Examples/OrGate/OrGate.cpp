
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


#define T float
using MatType = MatrixCPU<T>;
using MatrixRef = typename MatrixManager<T, MatType>::MatrixRef;

int main()
{
    ThreadPool pool;
    MatrixManager<T, MatType>& inst = MatrixManager<T, MatrixCPU<T>>::Instance();
    const uint32_t batchSize = 1;

    // Build Training Data
    ////////////////////////
    std::vector<MatrixRef> inputs;

    // 2xBatchsize matrix.
    // We set all 4 combinations of inputs
    auto newInput0 = inst.AllocateMatrix( {2,batchSize}, "Input");
    newInput0.get()->SetValue(0, 0, 0.0); // 0, 0
    newInput0.get()->SetValue(1, 0, 0.0);
    inputs.push_back(newInput0);

    auto newInput1 = inst.AllocateMatrix( {2,batchSize}, "Input");
    newInput1.get()->SetValue(0, 0, 1.0); // 1, 0
    newInput1.get()->SetValue(1, 0, 0.0);
    inputs.push_back(newInput1);

    auto newInput2 = inst.AllocateMatrix( {2,batchSize}, "Input");
    newInput2.get()->SetValue(0, 0, 0.0); // 0, 1
    newInput2.get()->SetValue(1, 0, 1.0);
    inputs.push_back(newInput2);

    auto newInput3 = inst.AllocateMatrix( {2,batchSize}, "Input");
    newInput3.get()->SetValue(0, 0, 1.0); // 1, 1
    newInput3.get()->SetValue(1, 0, 1.0);
    // Add to the array of inputs (Note: we have only one input in this demo)
    inputs.push_back(newInput3);

    // Use the input to build the outputs.
    std::vector<MatrixRef> outputs;
    uint32_t counter = 0u;
    for (auto& v : inputs)
    {
        auto newOutput = inst.AllocateMatrix( {2,batchSize}, "TargetOutput_" + std::to_string(counter));
        counter++;
        Fill(newOutput.get(), 0.0f);
        for (uint32_t j = 0; j < batchSize; j++)
        {
            if (v->GetValue(0, j) == 1.0 || v->GetValue(1, j) == 1.0)
            {
                newOutput.get()->SetValue(0, j,   1.0);
                newOutput.get()->SetValue(1, j,  -1.0);
            }
            else
            {
                newOutput.get()->SetValue(0, j,  -1.0);
                newOutput.get()->SetValue(1, j,   1.0);
            }
        }
        outputs.push_back(newOutput);
    }

    // Setup Optimiser
    //////////////////
    AdamOptimiser<T, MatType>* adamopt = new AdamOptimiser<T, MatType>(0.01);

    // Create Layers and Model
    //////////////////////////
    // One Dense layer with two inputs and two outputs.
    const uint32_t outputsize = outputs[0]->GetDimsX();
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
    m.SetOptimiser(adamopt);
    m.Init();
    
    // Run Training
    int count = 0;
    while (count < 100000)
    {
        for(uint32_t i = 0; i < inputs.size(); i++)
        {
            m.Run(inputs[i].get(), outputs[i].get());
            if(count % 1000 == 0)
            {
                LOG_INFO() << "Count: " << count;
                //LOG_INFO() << "Input" << i << ": ";
                //LOG_INFO() << inputs[i]->GetString();
                //LOG_INFO() << m.GetOutput()->GetString();
                //LOG_INFO() << outputs[i]->GetString();
                LOG_INFO() << m.GetFinalError()->GetString();
                //LOG_INFO() << denseData0->GetMatrix<MatType>("Weights")->GetString();
                //LOG_INFO() << denseData0->GetMatrix<MatType>("Bias")->GetString();
            }
            count++;
        }
    }

    return 0;
}