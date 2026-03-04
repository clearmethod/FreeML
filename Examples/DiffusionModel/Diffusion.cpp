
#include <cstdint>
#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <ModelLibrary/Model.h>
#include <LayerLibrary/Layers.h>
#include <OptimiserLibrary/Optimisers.h>
#include <LossLibrary/Losses.h>
#include <ToolsLibrary/Tools.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/CPU/MatrixCPU.h>

#ifdef _WIN32
    #include <MatrixLibrary/GPU/DirectX11/DirectX11Manager.h>
    #include <MatrixLibrary/GPU/DirectX11/MatrixDX11.h>
#endif

#include <Loading/SaveLoader.h>

#include "DiffusionHelpers.h"

#define T float

#ifdef _WIN32
using MatType = MatrixDX11<T>;
#else
using MatType = MatrixCPU<T>;
#endif

using MatrixRef = typename MatrixManager<T, MatType>::MatrixRef;
using diffusion_example::ApplyForwardNoise;
using diffusion_example::ComputeCosineNoiseSchedule;
using diffusion_example::ComputeNoiseSchedule;
using diffusion_example::TimeStepSample;
using diffusion_example::DDPMStep;
using diffusion_example::FillTimestepConditioning;
using diffusion_example::GenerateCheckerboard;
using diffusion_example::kTimeConditionChannels;
using diffusion_example::NoiseSchedule;
using diffusion_example::SampleGaussian;
using diffusion_example::PrintMatrixToTerminal;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct DiffusionConfig
{
    uint32_t image_ch        = 1u;
    uint32_t image_size      = 32u;
    uint32_t latent_ch       = 4u;
    uint32_t latent_size     = image_size / 2u;   // image_size / 2  (one stride-2 VAE encoder stage)
    uint32_t vae_hidden      = 32u;   // intermediate channel count in VAE
    uint32_t unet_C          = 32u;   // base channel multiplier for U-Net
    uint32_t T_steps         = 10u;    // smaller schedule → better per-step coverage per iteration
    uint32_t checker_tile    = 4u;    // checkerboard tile size in pixels
    uint32_t vae_train_iters = 5000u;   // phase 1: train VAE on image reconstruction
    uint32_t den_train_iters = 5000u;  // phase 2: train denoiser with frozen VAE encoder
    float    lr              = 1e-3f;
    uint32_t save_interval   = 1000u;
    uint32_t log_interval    = 100u;
    std::string vae_name      = "DiffVAE_v2";   // combined encoder + decoder
    std::string denoiser_name = "DiffDenoiser_v5";
};

// ---------------------------------------------------------------------------
// DDPM noise schedule  (linear beta schedule)
// ---------------------------------------------------------------------------

static void Sample(Model<T, MatType>*         vae,
                   Model<T, MatType>*         denoiser,
                   const DiffusionConfig&     cfg,
                   const NoiseSchedule&       sched,
                   MatrixManager<T, MatType>& inst);

// ---------------------------------------------------------------------------
// Model builders
// ---------------------------------------------------------------------------

// Combined VAE: encoder (VariationalAutoencoder) chained to
// decoder (VariationalAutoencoder_Decode) via AddDependency.
static std::unique_ptr<Model<T, MatType>> BuildOrLoadVAE( const DiffusionConfig& cfg, MatrixManager<T, MatType>&)
{
    auto model = std::make_unique<Model<T, MatType>>();
    if (!model->Load(cfg.vae_name))
    {
        // Encoder
        std::vector<ConvSettings> encStages = {
                                                {cfg.image_ch, cfg.vae_hidden, Dims3D(3u, 3u, 1u), 2u, 1u, 1u, true}
                                              };
        auto* enc = new VariationalAutoencoder<T, Relu<T>, MatType>();
        enc->SetName("vae_enc");
        auto* encBlob = InitVariationalAutoencoderBlob<T, MatType, Relu<T>>(encStages, cfg.latent_ch, true, 1e-4f);
        model->AddLayer(enc, encBlob);

        // Decoder
        std::vector<ConvTransposeSettings> decStages = {
                                                        {cfg.latent_ch,  cfg.vae_hidden, Dims3D(3u, 3u, 1u), 2u, 1u, 1u, 1u, true},
                                                        {cfg.vae_hidden, cfg.image_ch,   Dims3D(3u, 3u, 1u), 1u, 1u, 0u, 1u, true}
                                                       };
        auto* dec     = new VariationalAutoencoder_Decode<T, Identity<T>, MatType>();
        dec->SetName("vae_dec");
        auto* decBlob = InitVariationalAutoencoderDecodeBlob<T, MatType>(decStages, true);
        model->AddLayer(dec, decBlob);
        model->AddDependency(dec, 0, enc, 0);

        model->SetFinalOutputLayer(dec);
        model->SetExpectedInputDims({cfg.image_size, cfg.image_size, cfg.image_ch});
        model->SetName(cfg.vae_name);
        model->SetOptimiser(new AdamWOptimiser<T, MatType>(cfg.lr, 0.9f, 0.999f, 1e-8f, 1e-4f));
        model->SetLoss(new MeanSquaredError<T, MatType>());
        model->Init();
    }
    else
    {
        model->Init();
    }
    return model;
}

// Denoiser U-Net
// Basically a single U-net!
static std::unique_ptr<Model<T, MatType>> BuildOrLoadDenoiser( const DiffusionConfig& cfg, MatrixManager<T, MatType>& inst)
{
    const uint32_t C     = cfg.unet_C;
    const uint32_t in_ch = cfg.latent_ch + kTimeConditionChannels;

    auto model = std::make_unique<Model<T, MatType>>();
    if (!model->Load(cfg.denoiser_name))
    {
        std::vector<ConvSettings> encStages =   {
                                                 {in_ch,         cfg.latent_ch, Dims3D(3u, 3u, 1u), 1u, 1u, 1u, true, "relu"},
                                                 {cfg.latent_ch, C * 4,         Dims3D(3u, 3u, 1u), 2u, 1u, 1u, true, "relu"},
                                                 {C * 4,         C * 8,         Dims3D(3u, 3u, 1u), 2u, 1u, 1u, true, "relu"},
                                                };
        std::vector<ConvTransposeSettings> decStages = {
                                                        {C * 8, C * 4,         Dims3D(3u, 3u, 1u), 2u, 1u, 1u, 1u, true, "relu"},
                                                        {C * 4, cfg.latent_ch, Dims3D(3u, 3u, 1u), 2u, 1u, 1u, 1u, true, "identity"},
                                                       };
        auto* unet = new UNet<T, Identity<T>, MatType>();
        unet->SetName("unet");
        auto* blob = InitUNetBlob<T, MatType>(encStages, decStages, true);
        model->AddLayer(unet, blob);

        model->SetFinalOutputLayer(unet);
        model->SetExpectedInputDims({cfg.latent_size, cfg.latent_size, in_ch});
        model->SetName(cfg.denoiser_name);
        model->SetOptimiser(new AdamWOptimiser<T, MatType>(cfg.lr, 0.9f, 0.999f, 1e-8f, 1e-4f));
        model->SetLoss(new MeanSquaredError<T, MatType>());
        model->Init();
    }
    else
    {
        model->Init();
    }
    return model;
}

// ---------------------------------------------------------------------------
// Training buffers
// Prealloacted buffer external to the model for use in training
// and inference.
// ---------------------------------------------------------------------------
struct TrainingBuffers
{
    MatrixRef image;     // [image_size, image_size, image_ch]
    MatrixRef latent;    // [latent_size, latent_size, latent_ch]
    MatrixRef eps;       // [latent_size, latent_size, latent_ch]  sampled noise
    MatrixRef noiseBank; // [latent_size, latent_size, latent_ch * 512] pre-generated Gaussian noise samples
    MatrixRef zt;        // [latent_size, latent_size, latent_ch]  noisy latent
    MatrixRef tChannel;  // [latent_size, latent_size, 4]          timestep features
    MatrixRef unetIn;    // [latent_size, latent_size, latent_ch+4]
};

static TrainingBuffers AllocateTrainingBuffers(const DiffusionConfig& cfg,
                                               MatrixManager<T, MatType>& inst)
{
    TrainingBuffers buf;
    buf.image    = inst.AllocateMatrix({cfg.image_size,  cfg.image_size,  cfg.image_ch},           "tr_image");
    buf.latent   = inst.AllocateMatrix({cfg.latent_size, cfg.latent_size, cfg.latent_ch},          "tr_latent");
    buf.eps      = inst.AllocateMatrix({cfg.latent_size, cfg.latent_size, cfg.latent_ch},          "tr_eps");
    buf.noiseBank= inst.AllocateMatrix({cfg.latent_size, cfg.latent_size, cfg.latent_ch * 512u},   "tr_noise_bank");
    buf.zt       = inst.AllocateMatrix({cfg.latent_size, cfg.latent_size, cfg.latent_ch},          "tr_zt");
    buf.tChannel = inst.AllocateMatrix({cfg.latent_size, cfg.latent_size, kTimeConditionChannels}, "tr_tchan");
    buf.unetIn   = inst.AllocateMatrix({cfg.latent_size, cfg.latent_size, cfg.latent_ch + kTimeConditionChannels}, "tr_unet_in");
    return buf;
}

// ---------------------------------------------------------------------------
// Phase 1: Train VAE on image reconstruction
//   vae->Run(image, image) trains encoder + decoder end-to-end via MSE loss.
// This is to give a grounding between latent space and expected output.
// so when we generate a latent representation it can be constructed into
// the target image.
// ---------------------------------------------------------------------------
static void TrainVAE(Model<T, MatType>*         vae,
                     const DiffusionConfig&     cfg,
                     TrainingBuffers&           buf,
                     MatrixManager<T, MatType>&)
{
    // Generate the checkerboard once — it's the same every iteration.
    GenerateCheckerboard(buf.image.get(), cfg.checker_tile);

    for (uint32_t iter = 0; iter < cfg.vae_train_iters; ++iter)
    {
        // target = input: reconstruction loss trains both encoder and decoder.
        vae->Run(buf.image.get(), buf.image.get(), iter % cfg.log_interval == 0);

        if (iter % cfg.log_interval == 0)
        {
            LOG_INFO() << "[VAE] iter " << iter << "  loss: " << vae->GetLastLoss();
            PrintMatrixToTerminal(vae->GetOutput());
        }

        if (iter % cfg.save_interval == 0 && iter > 0u)
            vae->Save();
    }
    vae->Save();
}

// ---------------------------------------------------------------------------
// Phase 2: Train denoiser with VAE encoder frozen (forward-only)
// ---------------------------------------------------------------------------
static void TrainDenoiser(Model<T, MatType>*         vae,
                          Model<T, MatType>*         denoiser,
                          const DiffusionConfig&     cfg,
                          const NoiseSchedule&       sched,
                          TrainingBuffers&           buf,
                          MatrixManager<T, MatType>& inst)
{
    // Grab the encoder sub-layer so we can read its latent output after a
    // forward-only VAE pass without going through the full model output.
    auto* encLayer = static_cast<VariationalAutoencoder<T, Relu<T>, MatType>*>(
                         vae->GetLayer("vae_enc"));
    auto* encBlob  = vae->GetLayerBlob("vae_enc");
    if (!encLayer || !encBlob)
    {
        LOG_ERROR() << "Could not find vae_enc sub-layer.";
        return;
    }

    // Seed RNG
    std::mt19937 rng(42u);
    std::uniform_int_distribution<uint32_t> tDist(0u, cfg.T_steps - 1u);
    const uint32_t sampleElems = buf.eps.get()->GetElementCount();
    const uint32_t bankElems = buf.noiseBank.get()->GetElementCount();
    assert(bankElems >= sampleElems);
    std::uniform_int_distribution<uint32_t> noiseOffsetDist(0u, bankElems - sampleElems);

    // Encode once and keep for all denoiser iterations.
    // Use the sampled latent z (output index 0), which is the same latent
    // representation the decoder was trained to reconstruct from.
    GenerateCheckerboard(buf.image.get(), cfg.checker_tile);

    vae->Run(buf.image.get(), nullptr);
    MatrixRef latentRef = encLayer->GetOutput(encBlob, 0u);   // sampled z
    if (!latentRef.get())
    {
        LOG_ERROR() << "VAE encoder returned null sampled latent.";
        return;
    }
    Copy(buf.latent.get(), latentRef.get());

    // Pregenerate the input noise we are going to use.
    SampleGaussian(buf.noiseBank.get(), rng);

    for (uint32_t iter = 0; iter < cfg.den_train_iters; ++iter)
    {
        // Sample timestep and noise.
        const uint32_t tIdx = tDist(rng);
        const uint32_t noiseOffset = noiseOffsetDist(rng);
        CopyRange(buf.eps.get(), buf.noiseBank.get(), 0u, noiseOffset, sampleElems);
        MatType* epsMat = buf.eps.get();

        ApplyForwardNoise(buf.zt.get()->DataWrite(),
                          buf.latent.get()->DataRead(),
                          epsMat->DataRead(),
                          buf.zt.get()->GetElementCount(),
                          sched,
                          tIdx);

        // Build U-Net input: concat(zt, timestep_channel).
        FillTimestepConditioning(buf.tChannel.get(), tIdx, cfg.T_steps);
        ConcatZ(buf.unetIn.get(), buf.zt.get(), buf.tChannel.get());

        // Train denoiser to predict the noise eps.
        denoiser->Run(buf.unetIn.get(), epsMat, iter % cfg.log_interval == 0);

        // Log information at a fixed interval
        if (iter % cfg.log_interval == 0)
        {
            denoiser->PrintTimings();
            LOG_INFO() << "[Denoiser] iter " << iter << "  loss: " << denoiser->GetLastLoss() << "  t=" << tIdx;
            LOG_INFO() << "Rolling Loss: " << denoiser->GetRollingLoss();
            LOG_INFO() << denoiser->GetOutput()->GetString();
        }

        // Sample and save model at a different interval.
        if (iter % cfg.save_interval == 0 && iter > 0u)
        {
            denoiser->Save();
            Sample(vae, denoiser, cfg, sched, inst);
        }
    }

    // Save and sample on exit.
    denoiser->Save();
    Sample(vae, denoiser, cfg, sched, inst);
}

// ---------------------------------------------------------------------------
// DDPM sampling
// ---------------------------------------------------------------------------

static void Sample(Model<T, MatType>*         vae,
                   Model<T, MatType>*         denoiser,
                   const DiffusionConfig&     cfg,
                   const NoiseSchedule&       sched,
                   MatrixManager<T, MatType>& inst)
{
    // We want the decoder part of the VAE so we can turn the final latent back into an image.
    auto* decLayer = static_cast<VariationalAutoencoder_Decode<T, Identity<T>, MatType>*>(
                         vae->GetLayer("vae_dec"));
    auto* decBlob  = vae->GetLayerBlob("vae_dec");
    if (!decLayer || !decBlob)
    {
        LOG_ERROR() << "Could not find vae_dec sub-layer.";
        return;
    }

    // Fixed seed so sampling is repeatable while debugging.
    std::mt19937 rng(0u);

    // These are the working buffers used during sampling:
    // zt      = the current noisy latent
    // ztPrev  = the next, slightly cleaner latent
    // tChan   = the extra timestep information
    // unetIn  = current latent + timestep channels together
    auto ztRef     = inst.AllocateMatrix({cfg.latent_size, cfg.latent_size, cfg.latent_ch},       "smp_zt");
    auto ztNextRef = inst.AllocateMatrix({cfg.latent_size, cfg.latent_size, cfg.latent_ch},       "smp_zt_prev");
    auto tChanRef  = inst.AllocateMatrix({cfg.latent_size, cfg.latent_size, kTimeConditionChannels}, "smp_tchan");
    auto unetInRef = inst.AllocateMatrix({cfg.latent_size, cfg.latent_size, cfg.latent_ch + kTimeConditionChannels}, "smp_unet_in");

    // Start from pure random noise. This is the usual diffusion starting point.
    SampleGaussian(ztRef.get(), rng);
    const size_t latentElems = ztRef.get()->GetElementCount();

    // Walk through time from t=end to t=0.
    // Each step asks the denoiser: "what noise do you think is inside this latent?"
    // Then DDIMStep removes some of that noise.
    for (int32_t t = static_cast<int32_t>(cfg.T_steps) - 1; t >= 0; --t)
    {
        const uint32_t tIdx = static_cast<uint32_t>(t);

        // Fill the timestep channels so the network knows which denoising step it is on.
        FillTimestepConditioning(tChanRef.get(), tIdx, cfg.T_steps);

        // Join the current latent and the timestep info into one input tensor for the U-Net.
        ConcatZ(unetInRef.get(), ztRef.get(), tChanRef.get());

        // Run the denoiser. It predicts the noise inside the current latent.
        denoiser->Run(unetInRef.get(), nullptr);
        MatType* predEps = denoiser->GetOutput();
        if (!predEps)
        {
            LOG_ERROR() << "Denoiser returned null at t=" << t;
            return;
        }

        // Use the predicted noise to make the latent a bit cleaner than it was one step ago.
        TimeStepSample(ztNextRef.get()->DataWrite(),
                 ztRef.get()->DataRead(),
                 predEps->DataRead(),
                 latentElems,
                 sched, tIdx);

        // Print the current latent so we can watch the sampling process.
        LOG_INFO() << "[Sample] t = " << tIdx;
        PrintMatrixToTerminal(ztNextRef.get());

        // Move to the next step.
        Copy(ztRef.get(), ztNextRef.get());
    }

    // At the end, zt should now be z0: the final cleaned latent.
    // Send that latent through the VAE decoder to turn it into an image.
    LOG_INFO() << "Final latent range: min=" << Minimum(ztRef.get())
               << " max=" << Maximum(ztRef.get());
    decBlob->Set("Input_0", ztRef);
    decLayer->Forward(decBlob);
    MatType* decoded = decLayer->GetOutput(decBlob, 0u).get();
    if (decoded)
    {
        // Print some basic range info and then show the final image.
        LOG_INFO() << "Decoded range: min=" << Minimum(decoded)
                   << " max=" << Maximum(decoded);
        LOG_INFO() << "Sample complete. Image dims: "
                   << decoded->GetDimsX() << "x"
                   << decoded->GetDimsY() << "x"
                   << decoded->GetDimsZ();
        PrintMatrixToTerminal(decoded);
    }
    else
    {
        LOG_ERROR() << "VAE decoder returned null output";
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main()
{
    DiffusionConfig cfg;
    // Linear schedule: alpha_bar[9]≈0.044 keeps the t=9 DDIM step well-conditioned.
    // Cosine schedule needs T≈1000 — at T=10 it sends alpha_bar[T-1]→0 exactly,
    // making the denoiser's t=9 task degenerate (pure noise, x0 unrecoverable).
    const NoiseSchedule sched = ComputeNoiseSchedule(cfg.T_steps, 1e-4f, 0.5f);

    MatrixManager<T, MatType>& inst = MatrixManager<T, MatType>::Instance();

    auto vae      = BuildOrLoadVAE(cfg, inst);
    auto denoiser = BuildOrLoadDenoiser(cfg, inst);

    LOG_INFO() << "VAE:\n"      << vae->GetString();
    LOG_INFO() << "Denoiser:\n" << denoiser->GetString();

    TrainingBuffers buf = AllocateTrainingBuffers(cfg, inst);

    // Phase 1: train VAE on image reconstruction.
    TrainVAE(vae.get(), cfg, buf, inst);

    // Phase 2: train denoiser with frozen VAE encoder.
    TrainDenoiser(vae.get(), denoiser.get(), cfg, sched, buf, inst);

    // Generate a sample.
    Sample(vae.get(), denoiser.get(), cfg, sched, inst);
    PressAnyKeyToContinue();

    return 0;
}


