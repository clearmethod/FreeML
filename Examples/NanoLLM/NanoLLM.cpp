
#include <cstdint>
#include <deque>
#include <cassert>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <filesystem>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include <memory>

#include <ModelLibrary/Model.h>
#include <LayerLibrary/Layers.h>
#include <OptimiserLibrary/Optimisers.h>
#include <LossLibrary/Losses.h>

#include <ToolsLibrary/Tools.h>
#include "LearningRateSchedule.h"
#include "TokenMetrics.h"
#include <chrono>

#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/CPU/MatrixCPU.h>

#ifdef _WIN32
    #include <MatrixLibrary/GPU/DirectX11/DirectX11Manager.h>
    #include <MatrixLibrary/GPU/DirectX11/MatrixDX11.h>
#endif

#ifdef __APPLE__
    #include <MatrixLibrary/GPU/Metal/MetalManager.h>
    #include <MatrixLibrary/GPU/Metal/MatrixMetal.h>
#endif

#include <Loading/SaveLoader.h>

#define T float
// Switch to CPU if not on Windows.
#if defined(_WIN32)
    using MatType = MatrixDX11<T>;
#elif defined(__APPLE__)
    using MatType = MatrixMetal<T>;
#else
    using MatType = MatrixCPU<T>;
#endif


using LayerT = Layer<T, MatType>;
using BlobT = Datablob<T, MatType>;
using MatrixRef = typename MatrixManager<T, MatType>::MatrixRef;

struct NanoLLMConfig
{
    uint32_t block_size      = 256u; // max context length (sequence length)
    uint32_t batch_size      = 8u;   // sequences processed together in one matrix
    uint32_t n_layer         = 6u;   // number of transformer blocks
    uint32_t n_head          = 6u;   // number of attention heads per block
    uint32_t n_embed         = 384u; // embedding/hidden dimension
    uint32_t iter_count      = 500000u; // optimizer updates per training cycle
    uint32_t gradient_accumulation_steps = 8u; // physical batches per update; 8 * 8 = 64 sequences
    float    dropout         = 0.2f;
    float    learning_rate   = 0.001f; // peak rate after warmup
    float    minimum_learning_rate = 0.0001f;
    uint64_t warmup_updates  = 100u;
    uint64_t decay_updates   = 5000u; // hold the minimum rate after this update
    float    gradient_clip_norm = 1.0f; // zero disables global norm clipping
    bool     selective_weight_decay = true; // exclude Dense biases and LayerNorm parameters
    bool     use_bias        = false;
    uint32_t decode_Interval = 1u;
    uint32_t save_interval   = 300u;
    bool     sync_layer_timings = false; // wait for GPU completion after each timed layer

    #ifdef _DEBUG
    std::string model_name = "NanoLLM_DEBUG";
    #else
    std::string model_name = "NanoLLM_REL";
    #endif
};

static NanoLLMConfig DefaultConfig()
{
    return NanoLLMConfig{};
}

static bool ValidateConfig(const NanoLLMConfig& cfg)
{
    bool ok = true;
    if (cfg.block_size == 0u || cfg.batch_size == 0u || cfg.n_layer == 0u || cfg.n_head == 0u || cfg.n_embed == 0u)
    {
        LOG_ERROR() << "Invalid config: zero-sized dimension(s).";
        ok = false;
    }
    if (cfg.n_head != 0u && (cfg.n_embed % cfg.n_head) != 0u)
    {
        LOG_ERROR() << "Invalid config: n_embed must be divisible by n_head.";
        ok = false;
    }
    if (cfg.gradient_accumulation_steps == 0u)
    {
        LOG_ERROR() << "Invalid config: gradient_accumulation_steps must be positive.";
        ok = false;
    }
    if (!std::isfinite(cfg.gradient_clip_norm) || cfg.gradient_clip_norm < 0.0f)
    {
        LOG_ERROR() << "Invalid config: gradient_clip_norm must be finite and nonnegative.";
        ok = false;
    }
    if (!NanoLLMLearningRateSchedule{cfg.learning_rate, cfg.minimum_learning_rate,
                                    cfg.warmup_updates, cfg.decay_updates}.IsValid())
    {
        LOG_ERROR() << "Invalid learning-rate schedule: require finite rates, "
                    << "0 < minimum <= peak, and decay_updates > warmup_updates.";
        ok = false;
    }
    assert(cfg.n_head != 0u && (cfg.n_embed % cfg.n_head) == 0u);
    return ok;
}

struct Vocab
{
    std::vector<char> unique;
    std::vector<uint32_t> ids;
    std::map<char, uint32_t> toId;
    std::map<uint32_t, char> toChar;
};

static std::vector<char> LoadTextFile(const std::string& path)
{
    std::vector<char> chars;
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        LOG_ERROR() << "Failed to open file" << path;
        return chars;
    }

    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    for (unsigned char ch : contents)
    {
        chars.push_back(ch);
    }
    return chars;
}

static Vocab BuildVocab(const std::vector<char>& chars)
{
    Vocab vocab;
    vocab.unique = chars;
    std::sort(vocab.unique.begin(), vocab.unique.end());
    vocab.unique.erase(std::unique(vocab.unique.begin(), vocab.unique.end()), vocab.unique.end());

    const uint32_t vocab_size = static_cast<uint32_t>(vocab.unique.size());
    vocab.ids.reserve(chars.size());
    for (uint32_t i = 0u; i < vocab_size; ++i)
    {
        vocab.toId[vocab.unique[i]] = i;
        vocab.toChar[i] = vocab.unique[i];
    }
    for (unsigned char ch : chars)
    {
        auto it = vocab.toId.find(static_cast<char>(ch));
        vocab.ids.push_back(it == vocab.toId.end() ? 0u : it->second);
    }

    LOG_INFO() << "Vocab size:" << vocab_size;
    LOG_INFO() << "All chars: " << std::string(vocab.unique.begin(), vocab.unique.end());
    return vocab;
}

std::vector<uint32_t> encode(std::string _str, std::map<char, uint32_t>& charToUint)
{
    std::vector<uint32_t> out;
    out.reserve(_str.length());
    for(auto ch : _str)
    {
        out.push_back(charToUint[ch]);
    }
    return out;
}

std::vector<char> decode(std::vector<uint32_t> _data, std::map<uint32_t, char>& uintToChar)
{
    std::vector<char> out;
    out.reserve(_data.size());
    for(auto ch : _data)
    {
        out.push_back(uintToChar[ch]);
    }
    return out;
}

static std::unique_ptr<Model<T, MatType>> BuildOrLoadModel(const NanoLLMConfig& cfg,
                                                                   const Vocab& vocab,
                                                                   MatrixManager<T, MatType>& inst)
{
    const uint32_t vocab_size = static_cast<uint32_t>(vocab.unique.size());

    std::vector<TransformerBlockLayer<T, MatType>*> blocks;
    blocks.reserve(cfg.n_layer);

    auto model = std::make_unique<Model<T, MatType>>();
    if(!model->Load(cfg.model_name))
    {
        // Init Token Embedding
        auto* emebeddingToken = new Embedding<T, MatType>();
        emebeddingToken->SetName("embedding_token");
        auto* emebeddingToken_blob = InitEmbeddingBlob<T, MatType>(vocab_size, cfg.n_embed, cfg.block_size, cfg.batch_size, true);
        model->AddLayer(emebeddingToken, emebeddingToken_blob);

        // Init Positional Embedding
        auto* emebeddingPos = new Embedding<T, MatType>();
        emebeddingPos->SetName("embedding_pos");
        auto* emebeddingPos_blob = InitEmbeddingBlob<T, MatType>(cfg.block_size, cfg.n_embed, cfg.block_size, cfg.batch_size, true);
        model->AddLayer(emebeddingPos, emebeddingPos_blob);

        // Init Add: Add positional to token embedding.
        auto* addlayer = new AddLayer<T, MatType>();
        addlayer->SetName("token_pos_add");
        auto* addlayerblob = InitAddBlob<T, MatType>(true);
        model->AddLayer(addlayer, addlayerblob);
        model->AddDependency(addlayer, 0, emebeddingToken, 0);
        model->AddDependency(addlayer, 1, emebeddingPos, 0);

        // Init Dropout layer: (Only active in training).
        auto* dropoutLayer = new DropoutLayer<T, MatType>();
        dropoutLayer->SetName("embedding_dropout");
        auto* dropoutLayer_blob = InitDropoutBlob<T, MatType>(cfg.dropout, true);
        model->AddLayer(dropoutLayer, dropoutLayer_blob);
        model->AddDependency(dropoutLayer, 0, addlayer, 0);

        // Add all the transformer blocks
        for (uint32_t i = 0; i < cfg.n_layer; ++i)
        {
            auto* block = new TransformerBlockLayer<T, MatType>();
            block->SetName("block_" + std::to_string(i));
            const float residProjScale = 1.0f / std::sqrt(2.0f * static_cast<float>(cfg.n_layer));
            auto* blob = InitTransformerBlockBlob<T, MatType>(cfg.n_embed, cfg.n_head, cfg.dropout, cfg.block_size,
                                                                      cfg.batch_size, true, cfg.use_bias, residProjScale);
            model->AddLayer(block, blob);

            // Set dependency to previous block or the layer before blocks.
            if(i != 0)
                model->AddDependency(block, 0, blocks[i - 1], 0);
            else
            {
                model->AddDependency(block, 0, dropoutLayer, 0);
            }
            blocks.push_back(block);
        }

        // Init: Transformer output layer normalisation.
        auto* finalLayerNorm      = new LayerNorm<T, MatType>();
        finalLayerNorm->SetName("ln_f");
        auto* finalLayerNorm_blob = InitLayerNormBlob<T, MatType>(true, true, cfg.use_bias);
        model->AddLayer(finalLayerNorm, finalLayerNorm_blob);
        model->AddDependency(finalLayerNorm, 0, blocks.back(), 0);

        // Init final dense layer
        auto* finalDense      = new Dense<T, Identity<T>, MatType, true>();
        finalDense->SetName("lm_head");
        auto* finalDense_blob = InitDenseBlob<T, MatType>(vocab_size, cfg.n_embed, cfg.block_size, true, true, cfg.use_bias, true);

        // Tie embedding and final dense weights.
        MatType* finalDense_blobWeight      = finalDense_blob->GetMatrix<MatType>("Dense_Weights");
        MatType* emebeddingToken_blobWeight = emebeddingToken_blob->GetMatrix<MatType>("Embedding_Weights");
        if (!finalDense_blobWeight || !emebeddingToken_blobWeight)
        {
            LOG_WARNING() << "Failed to tie lm_head weights: missing Dense_Weights or Embedding_Weights.";
        }
        else
        {
            finalDense_blob->Set("Dense_Weights", inst.GetHandle(emebeddingToken_blobWeight));
            inst.RemoveMatrix(finalDense_blobWeight);
            LOG_INFO() << "Tied lm_head Dense_Weights to embedding weights.";
        }

        model->AddLayer(finalDense, finalDense_blob);
        model->AddDependency(finalDense, 0, finalLayerNorm, 0);
        model->SetFinalOutputLayer(finalDense);

        const float weightDecay = 0.01f;
        auto* adamopt = new AdamWOptimiser<T, MatType>(cfg.learning_rate, 0.9f, 0.99f, 1e-8f, weightDecay);
        model->SetOptimiser(adamopt);
        model->SetLoss(new CategoricalCrossEntropyWithLogits<T, MatType>());

        model->SetExpectedInputDims(MatrixShape::Sequence(cfg.batch_size,1u,cfg.block_size));
        model->SetName(cfg.model_name);
        model->Init();
    }
    else
    {
        model->SetExpectedInputDims(MatrixShape::Sequence(cfg.batch_size,1u,cfg.block_size));
        model->Init();
        PressAnyKeyToContinue();

        // If you want to change the model after it is loaded.
        // You do it here:
        // EXAMPLE:
        //      static_cast<AdamWOptimiser<T, MatType>*>(model->GetOptimiser())->SetLearningRate(0.00001f);
        // OR:
        //      model->GetLayerBlob("embedding_dropout")->Set("Probability", 0.00f);
        // 
    }

    model->SetGradientAccumulationSteps(cfg.gradient_accumulation_steps);
    model->SetGradientClipNorm(cfg.gradient_clip_norm);
    model->SetSelectiveWeightDecay(cfg.selective_weight_decay);
    model->SetTimingSynchronization(cfg.sync_layer_timings);
    return model;
}

struct TrainingMatrices
{
    MatrixRef posInput;
};

static TrainingMatrices AllocateTrainingMatrices(Model<T, MatType>* model,
                                               const NanoLLMConfig& cfg,
                                               MatrixManager<T, MatType>& inst)
{
    TrainingMatrices tensors;
    tensors.posInput = inst.AllocateMatrix(MatrixShape::Sequence(cfg.batch_size,1u,cfg.block_size), "Pos_Input");
    for (uint32_t batch=0;batch<cfg.batch_size;++batch)
        for (uint32_t t=0;t<cfg.block_size;++t)
            tensors.posInput->SetValue(batch,0u,0u,t,static_cast<T>(t));

    auto* emebeddingPos      = model->GetLayer("embedding_pos");
    auto* emebeddingPos_blob = model->GetLayerBlob("embedding_pos");
    if (emebeddingPos && emebeddingPos_blob)
    {
        emebeddingPos->SetInput(emebeddingPos_blob, tensors.posInput.get(), 0u);
    }

    return tensors;
}

static void TrainLoop(Model<T, MatType>*            model,
                      const NanoLLMConfig&          cfg,
                      const Vocab&                  vocab,
                      MatrixManager<T, MatType>&    inst)
{
    const auto&     uint32Data = vocab.ids;
    if (uint32Data.size() <= cfg.block_size)
    {
        LOG_WARNING() << "Not enough data to build a training window.";
        return;
    }

    // Preload full corpus once and copy sampled windows into dedicated train buffers.
    auto corpusRef = inst.AllocateMatrix(MatrixShape::Matrix(1u,static_cast<uint32_t>(uint32Data.size())), "Corpus_Tokens");
    MatType* corpus = corpusRef.get();
    assert(corpus);
    T* corpusPtr = corpus->DataWrite();
    for (uint32_t i = 0; i < static_cast<uint32_t>(uint32Data.size()); ++i)
    {
        corpusPtr[i] = static_cast<T>(uint32Data[i]);
    }

    auto inputBufferRef  = inst.AllocateMatrix(MatrixShape::Sequence(cfg.batch_size,1u,cfg.block_size), "Input_Buffer");
    auto targetBufferRef = inst.AllocateMatrix(MatrixShape::Sequence(cfg.batch_size,1u,cfg.block_size), "Target_Buffer");
    MatType* inputBuffer = inputBufferRef.get();
    MatType* targetBuffer = targetBufferRef.get();
    assert(inputBuffer);
    assert(targetBuffer);

    // Random number for picking a start position within the corpus for training.
    const uint32_t maxStart       = static_cast<uint32_t>(uint32Data.size() - cfg.block_size - 1u);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> startDist(0u, maxStart);

    model->SetTimingSynchronization(cfg.sync_layer_timings);

    auto* adam = dynamic_cast<AdamWOptimiser<T, MatType>*>(model->GetOptimiser());
    if (!adam)
    {
        LOG_ERROR() << "NanoLLM learning-rate scheduling requires AdamW.";
        return;
    }
    const NanoLLMLearningRateSchedule schedule{cfg.learning_rate, cfg.minimum_learning_rate,
                                             cfg.warmup_updates, cfg.decay_updates};
    // Existing AdamW checkpoints retain this count, including older checkpoints.
    uint64_t update = adam->GetMaximumStepCount();
    LOG_INFO() << "Resuming learning-rate schedule after " << update << " optimizer updates.";

    model->SetRollingLossWindow(100u);
    const uint64_t sequencesPerUpdate=uint64_t(cfg.batch_size)*cfg.gradient_accumulation_steps;
    LOG_INFO() << "Training with " << cfg.batch_size << " sequences per physical batch, "
               << cfg.gradient_accumulation_steps << " accumulation steps, "
               << sequencesPerUpdate << " sequences (" << sequencesPerUpdate*cfg.block_size
               << " tokens) per optimizer update.";

    // Loop forever training.
    while(true)
    {
        for(uint32_t i = 0; i < cfg.iter_count; i++)
        {
            const auto updateStart=std::chrono::steady_clock::now();
            const float learningRate = schedule.AtUpdate(update);
            adam->SetLearningRate(learningRate);
            model->BeginLossBatch();
            for (uint32_t pass = 0; pass < cfg.gradient_accumulation_steps; ++pass)
            {
                for (uint32_t batch=0;batch<cfg.batch_size;++batch)
                {
                    const uint32_t start = (maxStart == 0u ? 0u : startDist(rng));
                    CopyRange(inputBuffer, corpus, batch*cfg.block_size, start, cfg.block_size);
                    CopyRange(targetBuffer, corpus, batch*cfg.block_size, start+1u, cfg.block_size);
                }
                model->Run(inputBuffer, targetBuffer);
            }
            ++update;

            // Read once, after every backward pass and the optimizer update
            // have been recorded. Every sequence has the same token count.
            const double batchLoss = model->EndLossBatch();
            const double updateSeconds=std::chrono::duration<double>(
                std::chrono::steady_clock::now()-updateStart).count();

            LOG_INFO() << "Iteration: " << i << " Optimizer updates: " << update
                       << " LR: " << learningRate << " Loss: " << batchLoss
                       << " Rolling loss: " << model->GetRollingLoss()
                       << " Update(ms): " << updateSeconds*1000.0
                       << " Tokens/s: " << double(sequencesPerUpdate*cfg.block_size)/updateSeconds;
            // Save after a complete optimizer update.
            if (i % cfg.save_interval == 0 && i > 0u)
            {
                model->Save();
            }

            // Output some example and stats
            if (i % cfg.decode_Interval == 0)
            {
                model->PrintTimings();
                MatType* output = model->GetOutput();
                if (output)
                {
                    double lastRunTime = model->GetLastRunTime_Seconds();
                    LOG_INFO() << (cfg.sync_layer_timings ? "Last microbatch time (s): " : "Last microbatch CPU call time (s): ") << lastRunTime;

                    // Dump Memory Information
                    LOG_INFO() << inst.GetString();

                    // Display the first sample in the physical batch.
                    float meanacc = 0.f;
                    static std::deque<float> accuracyHistory;
                    const std::string decodedIn = NanoLLM::DecodeTokenBatch(inputBuffer, vocab.unique,0u);
                    LOG_INFO() << "Decoded input batch 0: \n" << decodedIn;
                    const std::string decodedTarget = NanoLLM::DecodeTokenBatch(targetBuffer, vocab.unique,0u);
                    LOG_INFO() << "Decoded target batch 0: \n" << decodedTarget;
                    const std::string decodedOut = NanoLLM::DecodeTokenBatch(output,vocab.unique);
                    LOG_INFO() << "Decoded output batch 0: \n" << Tools::Logger::Colour(128, 255, 225) << decodedOut << Tools::Logger::ClearColour();

                    // Quick accuracy check against targets.
                    const uint64_t correct=NanoLLM::CountCorrectTokens(output,targetBuffer);
                    const float acc=static_cast<float>(correct)/static_cast<float>(uint64_t(cfg.block_size)*cfg.batch_size);
                    LOG_INFO() << "Batch mean token accuracy: " << acc;
                    accuracyHistory.push_back(acc);
                    if (accuracyHistory.size() > 200)
                        accuracyHistory.pop_front();
                    for (auto hisval : accuracyHistory)
                    {
                        meanacc += hisval;
                    }
                    meanacc /= accuracyHistory.size();
                    LOG_INFO() << "Mean token accuracy: " << Tools::Logger::Colour(uint32_t((1.0-meanacc)*255), uint32_t(meanacc*255), 64u) <<  meanacc << Tools::Logger::ClearColour();
                }
            }
        }
    }
}

int main()
{
    NanoLLMConfig cfg = DefaultConfig();
    if (!ValidateConfig(cfg))
    {
        return 0;
    }

    // Matrix manager instance for loading matrices.
    MatrixManager<T, MatType>& inst = MatrixManager<T, MatType>::Instance();

    // LOAD Data
    /////////////////////////
    auto chars = LoadTextFile("Data/input.txt");
    Vocab vocab = BuildVocab(chars);

    // LOAD Model
    /////////////////////////
    // Row-major tensors use [batch, channels, width, height].
    // Tokens/positions: [B,1,1,T]; embedding and transformer: [B,1,C,T];
    // vocabulary logits: [B,1,vocab,T]. Dense flattens B*T rows internally.
    // The graph is embedding_token + embedding_pos -> dropout -> blocks
    // -> final LayerNorm -> lm_head, without a flatten layer.

    auto model = BuildOrLoadModel(cfg, vocab, inst);

    TrainingMatrices tensors = AllocateTrainingMatrices(model.get(), cfg, inst);
    (void)tensors;

    // Log Starting State
    LOG_INFO() << model->GetString();
#ifdef _WIN32
    LOG_INFO() << DirectX11Manager::Instance()->GetString();
    LOG_INFO() << DirectX11Manager::Instance()->GetMemoryString();
#endif
    LOG_INFO() << inst.GetString();
    LOG_INFO() << model->GetModelInOut();

    // Run Training
    TrainLoop(model.get(), cfg, vocab, inst);

    return 0;
}
