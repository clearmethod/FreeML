# Matrix tests

The shared suite in `MatrixFunctions.cpp` checks every public operation in
`MatrixBase_Functions.h` and `MatrixBase_Mul.h`, including the corresponding
DirectX 11 and Metal overloads. It uses `float`, which is supported by all three
backends. Each case is registered separately with CTest.

From the repository root:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release --target MatrixTests
ctest --test-dir build -C Release -L matrix --output-on-failure -j 4
```

`MatrixTests` builds all backends available on the current platform. To build
and run just CPU tests:

```sh
cmake --build build --config Release --target MatrixTests_CPU
ctest --test-dir build -C Release -L CPU --output-on-failure
```

To build and run just Metal tests on macOS:

```sh
cmake --build build --target MatrixTests_Metal
ctest --test-dir build -L Metal --output-on-failure
```

On Windows (the configuration flags also work with Visual Studio generators):

```sh
cmake --build build --config Release --target MatrixTests_DX11
ctest --test-dir build -C Release -L DX11 --output-on-failure
```

Use `ctest --test-dir build -N` to list cases, or select one with a regular
expression, for example `-R '^Matrix.Metal.MulTransposeBoth$'`. CPU and GPU
tests can also be run directly from `build/testing`, passing the case name
(for example `./MatrixTests_Metal MulTransposeBoth`).

GPU shader files are copied automatically when the test target is built;
CTest sets the working directory to `build/testing`. GPU tests report
**Skipped** (exit code 77) if no device is available. Shader errors and incorrect
results fail tests. Library assertions remain enabled in Release configurations.

## Coverage

- Copy, partial copy (including a zero-length range), clear, scalar/matrix fill.
- Add, subtract, elementwise multiplication, scalar/matrix scaling and scale-add,
  broadcasting by column/row, and threaded overloads. Elementwise operations
  include in-place cases and multiple Z planes.
- Mean squared error, triangular masks, row/column/spatial sums, minimum/maximum.
- Transpose, horizontal/vertical/Z concatenation, both vector concatenation
  overloads, batched gather/scatter, repeated-index scatter-add, QKV split/merge.
- Mapping functions, threaded mapping, custom binary mapping, ReLU/GELU and
  derivatives; softmax with extreme inputs and finite-difference backward checks.
- Layer normalization with/without bias and saved normalized values; backward
  gradients checked independently using finite differences and parameter sums.
- Fused GELU backward: derivative-times-gradient parity, independent finite
  differences, negative/zero/positive inputs, long rows, slice offsets, unchanged
  inputs/padding, queued GPU compute, and DenseGelu scratch preservation.
- Fused attention score gradients across six planes: causal masking, scaling,
  offset inputs/output/scalar, exact upper-triangle zeros, and queued GPU work.
  Results are checked against the original three operations and independent math.
- All four transpose modes for multiplication and batched strided multiplication,
  column-major strides, larger non-square matrices, partial blocks, raw-pointer
  and matrix multiplication helpers, explicit threads and thread pools.
- Explicit batch strides: interleaved heads, padded planes, column-major storage,
  broadcast reads, nonzero offsets, and both scalar and tiled Metal paths.
  Tests inspect full backing buffers for untouched padding and unchanged inputs.
- Compute → consecutive GPU copies → compute ordering, copies remaining in one
  Metal command buffer, and waits after recorded copy-only or submitted work.
- Slice copies between different-sized buffers and within a buffer, mapped and
  GPU paths; removing a borrowed view while another reference remains alive.
- Convolution and transposed convolution, forward/backward, with multiple input
  channels, an output-channel offset, optional bias/gradients, stride, dilation,
  padding, and accumulation into existing gradients.
- Adam/AdamW parameters and moments over several updates; reparameterization,
  clamped log-variance, saved random noise, and backward gradients.
- Matrix value, row/column and raw-data access; Z-slice reads and writes with
  nonzero buffer offsets.

Expected results come from fixed values, independent mathematical references,
or finite differences, rather than treating the CPU implementation as an oracle.
Output buffers start with NaNs so an operation that silently does nothing fails.
Random sampling uses broad statistical bounds and checks each latent value
against its saved noise.

The suite calls operations with concrete backend types. Where a backend has no
specialized overload (for example Metal convolution), this exercises the shared
implementation through that backend's CPU mapping. Internal dispatch, buffer
synchronization, float packing and alignment helpers are exercised through their
public callers. The suite covers operations and representative paths; it is not
an exhaustive test of every shape, invalid input, optional argument or data type.

Set `-DBUILD_TESTING=OFF` to disable test targets and registration.

## Layer tests

`Layers.cpp` tests all 13 concrete layer classes exported by `Layers.h` using
`float` on CPU, Metal (macOS), and DirectX 11 (Windows). `MatrixBase` is an
interface and `MatrixDynamicUnset` is a placeholder, so neither is a runnable
backend. Each of the 29 fixtures has separate Forward and Backward CTests.

The shared fixtures and correctness checks live in `LayerTestSupport.h`.
The [runtime evaluators](../eval/README.md) use the same coverage and report
completed forward/backward wall times for each fixture.

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release --target MatrixTests LayerTests
ctest --test-dir build -C Release -L layer --output-on-failure -j 4
```

Select one backend or pass with a regular expression:

```sh
ctest --test-dir build -C Release -R '^Layer.Metal.' --output-on-failure
ctest --test-dir build -C Release -R '^Layer.*Backward$' --output-on-failure
```

Use `LayerTests_CPU`, `LayerTests_Metal`, or `LayerTests_DX11` to build one
backend. A case can also run directly from `build/testing`, for example
`./LayerTests_Metal TransformerBlock Backward` (append `.exe` on Windows).

Coverage includes:

- ActivationLayer: identity, ReLU, optimized ReLU, leaky ReLU, sigmoid, tanh, GELU.
- Dense: ordinary and transposed weights, bias, identity and GELU activation.
- AddLayer: matching shapes and broadcasting over Z, including reduced input gradients.
- FlattenCopyLayer: values and flattened output shape, backward reshape.
- DropoutLayer: zero probability, training mask/scaling, inference identity.
- Embedding: repeated IDs across two batches, accumulated weight gradients,
  zero gradients for discrete token IDs.
- LayerNorm: multiple rows/batches, with and without beta; training-only inverse
  standard deviation caching, cache resizing and refresh, and inference without
  cache allocation or writes. Matrix tests also cover offset cache slices,
  constant rows, feature counts above one threadgroup, and backward after the
  original input has been overwritten.
- Conv2D and Conv2DTranspose: multiple input/output channels, nontrivial kernels,
  bias, transposed-convolution stride.
- CausalSelfAttentionLayer: two heads, QKV and projection weights/biases, causal prefix.
- TransformerBlockLayer: norms, attention, GELU MLP, both residual paths, causal prefix.
- UNet: two encoder/decoder stages, skip connection and accumulated backward paths.
- VariationalAutoencoder: encoder and both heads, sampled latent algebra, fixed-noise backward.
- VariationalAutoencoder_Decode: two transposed-convolution stages.

Every forward fixture has an independent scalar mathematical reference; deterministic
GPU outputs also match CPU results. Backward checks use central finite differences
for every input element and every trainable weight/bias, then repeat backward
to check gradients are overwritten. Embedding IDs are discrete, so only its
weights have numerical derivatives. Dropout backward uses a fixed nontrivial
mask; VAE derivatives use fixed nonzero noise and KL weight zero, isolating the
reconstruction path. Sampling/masks are separately checked in forward tests.
Attention also covers six heads, shared-storage resizing, and dropout gradients
against an independent reference using the saved sampled masks.

`MatMul_Batched<Mode>(out, left, right, outputStrides, leftStrides, rightStrides)`
accepts `MatrixStrides{row, column, batch}` in elements. Matrix views have the same
Z dimension (batch count), including broadcast inputs. Use positive row/column
strides; a zero input batch stride broadcasts the same physical matrix. Output
addresses must be distinct, and backing storage must cover all strided accesses.
Existing `MatMul_Strided` calls retain their implicit plane strides.

Fixtures use supported layouts: Z is channels for convolution, while attention
and transformer use batch size one. This is representative coverage, not every
configuration, scalar type, or invalid input. Backend fallbacks through CPU
mapping are exercised where no GPU kernel exists. A GPU unavailable to the
process is reported as a skip, rather than a pass.

`ctest --test-dir build -R NanoLLMConvergence --output-on-failure` runs a
deterministically initialized, miniature NanoLLM training regression on each
available backend. It includes six-head attention, GELU, LayerNorm, tied
embedding/output weights, and AdamW. It must learn a fixed next-token sequence
within 100 updates and update each tied parameter exactly once per step.

## Explicit physical batching

`MatrixShape` storage tests cover four-axis row-major indexing, checked size
arithmetic, offset reset, scratch isolation, retained/nested/rebound views,
ownership-cycle rejection and an actual disk round trip after releasing the
original allocation. Model checkpoints also exercise tied weights and optimizer
state. No old-format checkpoint conversion is supported.

`FourDimensional*` matrix cases test all 16 reduction-axis combinations and
complementary singleton broadcasts; all four concatenation axes; all four matmul
transpose modes with independent batch/channel broadcasting, interleaved/padded
strides, offsets, sentinels and partial tiles; and normal/transposed convolution
with non-square spatial dimensions, multiple batches/channels, stride, dilation,
padding and output padding. Metal tests verify new operations remain queued.

`Model.<backend>.BatchedLayers` compares 26 deterministic layer variants at
physical batch sizes 1/2/3 against independent sample outputs, input gradients and
summed shared parameter gradients. This includes sequence and image composites.
Stochastic dropout has separate mask, forward/backward and evaluator checks.
`NanoLLMPhysicalBatchConvergence` trains a physical batch of three with four
accumulation steps. `BatchUpdateEquivalence` compares 12 independent passes
against four physical batches of three through clipped/selective AdamW updates,
including tied parameters, mean loss, optimizer moments and update counts.

```sh
ctest --test-dir build -R 'FourDimensional|MatrixShape|BatchedLayers|PhysicalBatch|BatchUpdateEquivalence|BatchedMatrices' --output-on-failure
```

The same tests are registered for CPU and the platform GPU backend. DX11 source
is included; running its tests requires Windows. Batched evaluator checks validate
real execution without imposing hardware-dependent performance thresholds.

`Model.<backend>.NanoLLMTokenMetrics` checks decode/accuracy across distinct
samples and offset views, including ties, invalid batches, mismatched target
shapes and unchanged input sentinels. Repeated diagnostic calls must leave the
matrix registry and owned-memory report unchanged.
