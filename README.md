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

Matrix function and layer forward/backward tests are available through CTest for CPU, DirectX 11 (Windows),
and Metal (macOS). See [testing/README.md](testing/README.md) for build commands
and coverage.

Per-layer forward/backward runtime evaluators are available in [eval](eval/README.md).
Build with `cmake --build build --target LayerEval`, then run
`./build/eval/LayerEval` for side-by-side timings across available matrix backends.

## Examples/HelloFreeML
A minimal end-to-end example that follows the Getting Started guide and shows how to create a model, wire layer dependencies, run training, and inspect loss/output logs.
Source: `Examples/HelloFreeML/HelloFreeML.cpp`

## Examples/OrGate & Examples/OrGateBatched
***The best place to start looking at how to use this library.***
A simple gate implemented with Dense layers to demonstrate how to assemble models and the input/training data.

## Examples/NanoLLM (*NanoGPT++*)
Metal cross-entropy keeps logits, targets and gradients on the GPU. Index targets
use a mean over valid rows (ignoring `-1`); one-hot targets preserve the existing
summed-loss convention. Requesting a loss value reads back only one scalar.

NanoLLM warms up its base learning rate for `warmup_updates` optimizer updates,
then follows cosine decay from `learning_rate` to `minimum_learning_rate` by
`decay_updates`, holding the minimum afterward. Defaults are 100 warmup updates,
5,000 decay updates, and rates of 0.001 and 0.0001. Accumulated sequence passes
share one rate. The schedule resumes from AdamW's checkpointed parameter step
counts and continues across training cycles; logs show the cumulative update
count and rate used. Schedule settings come from the current configuration.

NanoLLM reports `Loss` as the mean over all sampled sequences contributing to
the optimizer update. `Rolling loss` averages the last 100 completed batch
means (or all completed batches if fewer than 100). These are training losses;
the rolling window starts fresh when the process resumes a checkpoint.
Metal accumulates these sequence losses on the GPU and reads one batch mean
after the sequence passes and optimizer work have been recorded. Other models
can use `BeginLossBatch()` before their passes and `EndLossBatch()` afterward;
ending a loss batch does not flush accumulated gradients. CPU and other losses
use their existing scalar evaluation through the same interface.
Timing summaries show the final readback under `Batch Loss`, once per batch;
its wait includes completion of previously queued GPU work.
NanoLLM logs `Update(ms)` and `Tokens/s` after that completion boundary, covering
all accumulation passes and the optimizer update, excluding subsequent decoding,
printing and checkpoint saves. Asynchronous `Total Run`/layer timings measure CPU
calls and must not be used as completed training throughput. To attribute waits
to individual layers, set `NanoLLMConfig::sync_layer_timings = true`; this adds
completion waits and changes execution overlap.

Token decoding and accuracy read each matrix once and index its batches directly;
they do not create temporary GPU matrix views. `MatrixManager` reports owned tensor
buffer bytes rather than counting borrowed views as additional allocations.
Constant buffers, staging and backend caches are not included in that tensor total.


Dropout uses the shared `DropoutMask(matrix, probability, seed)` function.
Metal and DirectX11 generate binary keep masks directly on the GPU; CPU uses
the same integer hash. The seed is optional and defaults to a fresh seed per
call. An explicit seed reproduces the same logical mask across backends and
slice offsets. Forward applies the existing inverted-dropout scaling, and
backward reuses the saved mask without generating new random values.

`NanoLLMConfig::gradient_clip_norm` defaults to 1.0 (zero disables clipping).
The model clips the global L2 norm after combining shared parameter gradients
and averaging accumulated passes, before any optimizer step. Other models can
opt in with `SetGradientClipNorm`; clipping is disabled by default. The
norm calculation uses parallel reductions on Metal and DirectX11, with reusable
scratch buffers and a GPU-resident scale factor. CPU retains its double-precision
reduction and non-finite error check; the GPU path omits that error check and
performs no gradient readback or norm-related completion wait. NanoLLM reapplies
the setting when loading a checkpoint.

NanoLLM enables `selective_weight_decay`: Dense biases and LayerNorm gamma/beta
use zero decay, while Dense and embedding weights use AdamW's configured decay
strength. The existing decay strength remains 0.01 for fresh models (or the saved
value for loaded models). Other models retain uniform decay unless they call
`SetSelectiveWeightDecay(true)`. Layers can override
`GetWeightDecayScale(blob, parameter)`; composite layers inherit their child
policies. Tied parameters must agree on the scale, and accumulation keeps the
policy through the optimizer update. NanoLLM reapplies the mode on checkpoint load.

`NanoLLMConfig::gradient_accumulation_steps` defaults to 8 sampled sequences
per optimizer update. `iter_count`, logging and save intervals count optimizer
updates; use 1 to restore one sequence per update. Gradients are averaged across
passes, so the learning rate does not need to be divided by this setting.

Other models can call `SetGradientAccumulationSteps(n)` before training.
`Run(input, target)` accumulates gradients; inference calls do not. Call
`FlushAccumulatedGradients()` to apply a partial window using its actual pass count,
or `ResetGradientAccumulation()` to discard it. Changing the window size or saving
flushes pending gradients; loading clears them. Accumulation settings are runtime
configuration and are not stored in checkpoints. Each pass
has equal weight, which matches NanoLLM's fixed-size training windows.

Set `NanoLLMConfig::sync_layer_timings = true` to include GPU completion in
each layer's printed forward/backward timing. This waits around every layer
and reduces CPU/GPU overlap. The default is `false`. Other models can enable
the same mode with `model->SetTimingSynchronization(true)` before `Run()`;
changing modes clears timing history. `PrintTimings()` reports the active mode.

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
- Validate DX11 physical batching on Windows.
- Switch DirectX11 to Slang, to increase GPU platform support. We would love to have the GPU driver in here too but that is a little outside of the scope...




## Physical batching and matrix shapes

Matrices use `MatrixShape{batch, channels, width, height}` and remain row-major:
`offset + (((batchIndex * channels + channel) * height + y) * width + x)`.
Use `MatrixShape::Matrix(width,height)`, `Sequence(batch,features,tokens)` or
`Image(batch,channels,width,height)` to state intent. Sequences have one channel;
attention heads use the channel axis internally. Image batches and channels are
independent. `GetShape()` replaces the former dimensional API.

All concrete layers support physical batches. Dense internally views batch/token
rows together while sharing weights. Conv2D and transposed convolution share
kernels across samples; parameter gradients sum across samples. Loss functions
supply averaging, and gradient accumulation averages microbatch gradients before
one optimizer update. NanoLLM's effective sequences per update are
`batch_size * gradient_accumulation_steps`. NanoLLM defaults to physical batch
8 with 8 accumulation steps, retaining 64 sequences (16,384 tokens at context
256) per optimizer update. Set the physical batch size in
`NanoLLMConfig`; set the model's expected input shape before `Init()`. Changing
a model's input dimensions requires another `Init()` to rebuild graph buffers.

`ReduceSum(out,input,MatrixAxes::Batch | MatrixAxes::Height)` reduces explicit
axes while retaining singleton dimensions. Binary broadcasting supports singleton
axes independently. `Concat(out,left,right,MatrixAxis::Channels)` preserves
samples. `MatMul_Batched` supports independent row/column/channel/batch strides;
strides are in elements. The legacy single-plane sum/raw multiplication helpers
keep their original contracts; use the explicit tensor APIs for batched work.

Managed `BatchView`, `ChannelView`, and `ReshapeView` retain backing storage.
Reshapes preserve element count and do not reorder data. Save files contain
BATCH/CHANNELS/WIDTH/HEIGHT fields; earlier checkpoint formats are unsupported.

Compare completed forward/backward time, throughput and tensor storage with:

```sh
cmake --build build --target Eval -j 4
./build/eval/LayerEval --layer TransformerSixHeads --batch-size 8 --scale 16
./build/eval/MatrixEval --function Mul --batch-size 8 --scale 64
```

See [batching-update.md](batching-update.md), [evaluation](eval/README.md) and
[batching test coverage](testing/README.md). CPU and Metal batching are validated
locally; DX11 runtime validation requires Windows.
