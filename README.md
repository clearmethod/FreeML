# FreeML

This is a repository which implements a programmable machine learning API with no external dependencies.

The goal is to let you set up and step through PyTorch-like models to understand how they work from first principles and to experiment with new layer types and data flow.

We call it FreeML because it is free from complex external library dependencies. Currently the only external dependencies are:
 - **C++ STL**
 - If you want to use the GPU on Windows: **DirectX11**. 

With this setup, you can see how a model can be constructed, linked and run with no secrets.

While the library aims for decent performance - its main goal is clarity. As a result, some obvious improvements (particularly with shared matrices) has been avoided.

It is also a side-project to support other work, so no guarantees can be given. Please feel free to submit any issues if something is not correct :)

# Features
- Model assembly, with saving and loading.
- Prewritten test layers including Transformers.
- Easy to use Memory manager and lifetime controls for permanent, temporary and shared matrices.
- Clear and easy to debug program flow with print functions to show the structure.

# Getting started

Models are ran through the `Model` class which is templated on the model data and matrix type, e.g `Model<float, MatrixCPU<float>>` will create a `Model` setup to handle `float` data hosted on the CPU.

```C++
Model<float, MatrixCPU<float>> m;
m.SetName("TestModel");
```

The `Model` is populated with layers and their data. Data is stored separately from layers in a `Datablob`, which holds named values and references required for layer execution. So this is how we create them:

```C++
// Creating a Datablob<float, MatrixCPU<float>> for a Dense layer using its helper.
auto* denseData0  = InitDenseBlob<float, MatrixCPU<float>>(4, 4, 1);
auto* denseLayer0 = new Dense<float, Tanh<float>, MatrixCPU<float>>();
m.AddLayer(denseLayer0, denseData0);

auto* denseData1  = InitDenseBlob<float, MatrixCPU<float>>(4, 4, 1);
auto* denseLayer1 = new Dense<float, Identity<float>, MatrixCPU<float>>();
m.AddLayer(denseLayer1, denseData1);
```

Dependencies are explicitly set using `Model`'s `AddDependency` function where you specify the destination layer and input index, and then the source layer and the output index. 

If we want to link the two layers defined above we would call:
```C++
m.AddDependency(denseLayer1, 0, denseLayer0, 0);
```

That will link the `denseLayer0` output `0` to the `denseLayer1` input `0` in our graph.

Once we have finished adding layers, we mark the final output:
```C++
m.SetFinalOutputLayer(denseLayer1);
```
**`Note:`** We can support multiple outputs for backprop, but for simplicity this has not been enabled in this release.

Next we will want to set an `Optimiser` and `Loss` function:

```C++
auto* optimiser = new AdamOptimiser<float, MatrixCPU<float>>(0.1f);
m.SetOptimiser(optimiser);

auto* lossfunc = new MeanSquaredError<float, MatrixCPU<float>>();
m.SetLoss(lossfunc);
```

Once that is complete, we are ready to initialise the model:
```C++
m.Init();
```
This will link the layers into the graph and attach inputs and outputs through the `Layer`'s `Datablobs`. It will also initialise any matrices required for the input and to run the model. Many layers do not have a hardcoded size and are dependent on their input size - this is where their memory is allocated once the input is known.

We can now run this model, but to do so we need to initialise input matrices. To do that we use the `MatrixManager`. Naming is optional but heavily encouraged to help with learning and debugging the neural networks. AllocateMatrix will return a `MatrixRef` which holds the pointer to the matrix and its lifetime manages the reference count for keeping a matrix in memory.

```C++
auto& inst = MatrixManager<float, MatrixCPU<float>>::Instance();
auto input = inst.AllocateMatrix( {4,1}, "Input");
auto target = inst.AllocateMatrix( {4,1}, "Output");

Fill(input.get(), 1.0f);
Fill(target.get(), 4.0f);
```

The `input` and `target` can now be used to run and train the model. Running a model with a target set will run the forward and backwards pass and update the trainable parameters.
```C++
for(uint32_t i = 0; i < 100; i++)
{
    m.Run(input.get(), target.get());
    if(i % 5 == 0)
        LOG_INFO() << "Loss: " << m.GetLastLoss();
}
LOG_INFO() << "Target: " << target->GetString();
LOG_INFO() << "Output: " << m.GetOutput()->GetString();
```
*`Note:`* LOG_INFO is part of ToolsLibrary/Logger.h. Logger is used throughout this codebase for controlling output.

Giving the output:
```js
[INFO] Loss: 15.9961
[INFO] Loss: 4.75732
[INFO] Loss: 0.0420133
[INFO] Loss: 0.945677
[INFO] Loss: 0.648032
[INFO] Loss: 0.00178843
[INFO] Loss: 0.224609
[INFO] Loss: 0.115273
[INFO] Loss: 0.00571344
[INFO] Loss: 0.0632787
[INFO] Loss: 0.00932224
[INFO] Loss: 0.0110953
[INFO] Loss: 0.0114167
[INFO] Loss: 0.000449729
[INFO] Loss: 0.00511664
[INFO] Loss: 0.00023535
[INFO] Loss: 0.00152373
[INFO] Loss: 0.00036303
[INFO] Loss: 0.000369761
[INFO] Loss: 0.00022793
[INFO] Target: Name: Output
Dims: (4, )
[ 4.000,4.000,4.000,4.000]

[INFO] Output: Name: Dense_Output
Dims: (4, )
[ 3.999,3.995,3.991,3.995]
```


If you want to view the model config and matrix memory usage you can use:
```C++
LOG_INFO() << m.GetString();
LOG_INFO() << m.GetModelInOut();
LOG_INFO() << inst.GetString();
```
Which produces output like:
```js
[INFO] Model: HelloFreeML
1: Dense0 [Dense]
2: Dense1 [Dense]

[INFO] Model: HelloFreeML
Dense0: (4, ) -> (4, )
Dense1: (4, ) -> (4, )

[INFO] MatrixManager
Total matrices: 30
Scratch cached: 0
Skeleton cached: 0
Total allocated: 864 bytes, 0.00 MB, 0.00 GB
```

From here you can experiment with linking different layers together, most layers have a helper function called Init{LayerName} which will initialise a supporting Datablob with its setup parameters.

# Demos

## Examples/HelloFreeML
A minimal end-to-end example that follows the Getting Started guide and shows how to create a model, wire layer dependencies, run training, and inspect loss/output logs.
Source: `Examples/HelloFreeML/HelloFreeML.cpp`

## Examples/OrGate & Examples/OrGateBatched
***The best place to start looking at how to use this library.***
A simple gate implemented with Dense layers to demonstrate how to assemble models and the input/training data.

## Examples/NanoLLM (*NanoGPT++*)
We have a C++ implementation of the Shakespeare model as structured in Karpathy's amazing [NanoGPT](https://github.com/karpathy/nanoGPT) Python project.
So you can train a GPT-2 lite model and see the output.

**Example**: @ 97% accuracy with direct decode.

Target:
```
t they must use in prayer.

ROMEO:
O, then, dear saint, let lips do what hands do;
They pray, grant thou, lest faith turn to despair.

JULIET:
Saints do not move, though grant for prayers' sake.

ROMEO:
Then move not, while my prayer's effect I take.
Thus
```
Output
```
t the  aast ase in prayer.

ROMEO:
O, then, dear saint, let lips do what hands do;
They pray, grant thou, lest faith turn to despair.

JULIET:
Saints do not move, though grant for prayers' sake.

ROMEO:
Then move not, while my prayer's effect I take.
Thus
```

# Library Modules

## Activations Library
- [GeLU](ActivationLibrary/Gelu.h) 
- [Identity](ActivationLibrary/Identity.h) 
- [LeakyReLU](ActivationLibrary/LeakyRelu.h) 
- [Relu](ActivationLibrary/Relu.h) 
- [ReLU (Opt)](ActivationLibrary/ReluOpt.h) 
- [Sigmoid](ActivationLibrary/Sigmoid.h) 
- [TanH](ActivationLibrary/Tanh.h)

## Layer Library

- [Activation](LayerLibrary/ActivationLayer.h) 
- [Add](LayerLibrary/AddLayer.h) 
- [CausalSelfAttention](LayerLibrary/CausalSelfAttentionLayer.h) 
- [Conv2D](LayerLibrary/Conv2D.h) 
- [Datablob](LayerLibrary/Datablob.h) 
- [Dense](LayerLibrary/Dense.h) 
- [Dropout](LayerLibrary/DropoutLayer.h) 
- [Embedding](LayerLibrary/Embedding.h) 
- [Flatten](LayerLibrary/FlattenCopyLayer.h) 
- [LayerNorm](LayerLibrary/LayerNorm.h) 
- [Transformer](LayerLibrary/TransformerBlock.h)

## Loss Library
- [Binary Cross Entropy](LossLibrary/BinaryCrossEntropy.h) 
- [CategoricalCrossEntropyWithLogits](LossLibrary/CategoricalCrossEntropyWithLogits.h) 
- [Direct Loss](LossLibrary/DirectLoss.h) 
- [DirectNormalisedLoss](LossLibrary/DirectNormalisedLoss.h) 
- [MeanSquaredError](LossLibrary/MeanSquaredError.h)
## Matrix Library
- [MatrixBase](MatrixLibrary/MatrixBase.h)
- [MatrixCPU](MatrixLibrary/CPU/MatrixCPU.h)
- [MatrixDX11](MatrixLibrary/GPU/DirectX11/MatrixDX11.h)
## Model Library
- [Model](ModelLibrary/Model.h)
## OptimiserLibrary
- [Adam](OptimiserLibrary/AdamOptimiser.h) 
- [AdamW](OptimiserLibrary/AdamWOptimiser.h) 
- [Basic](OptimiserLibrary/BasicOptimiser.h)
## ToolsLibrary
- [Logger](ToolsLibrary/Logger.h)
- [ThreadPool](ToolsLibrary/ThreadPool.h) 
- [Timer](ToolsLibrary/Timer.h)
- [Misc](ToolsLibrary/Tools.h)


# Future work
- Improve saving and loading of models.
- Improve CPU performance.
- Add in more layers.
- Add learning rate controllers.
- Add gradient accumulation.
- Add batch support in current transformer layers.
- Switch DirectX11 to Slang, to increase GPU platform support. We would love to have the GPU driver in here too but that is a little outside of the scope...




