
#include <cstdint>
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

#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/CPU/MatrixCPU.h>

#ifdef _WIN32
#include <MatrixLibrary/GPU/DirectX11/DirectX11Manager.h>
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11.h>
#endif

#include <Loading/SaveLoader.h>

#define T float
// Switch to CPU if not on Windows.
#ifdef _WIN32
using MatType = MatrixDX11<T>;//MatrixCPU<PRECISION>;
#else
using MatType = MatrixCPU<PRECISION>;//
#endif

using LayerT = Layer<T, MatType>;
using BlobT = Datablob<T, MatType>;
using MatrixRef = typename MatrixManager<T, MatType>::MatrixRef;

struct NanoLLMConfig
{
    uint32_t block_size      = 256u; // max context length (sequence length)
    uint32_t batch_size      = 1u;   // number of sequences per batch
    uint32_t n_layer         = 6u;   // number of transformer blocks
    uint32_t n_head          = 6u;   // number of attention heads per block
    uint32_t n_embed         = 384u; // embedding/hidden dimension
    uint32_t iter_count      = 50000u;
    float    dropout         = 0.0f;
    float    learning_rate   = 0.0001f;
    bool     use_bias        = false;
    uint32_t decode_Interval = 50u;
    uint32_t save_interval   = 300u;

    #ifdef _DEBUG
    std::string model_name = "NanoLLM_DEBUG";
    #else
    std::string model_name = "NanoLLM";
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
    if (cfg.batch_size != 1u)
    {
        LOG_ERROR() << "Invalid config: NanoLLM currently supports batch_size == 1 only.";
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

std::string DecodeMatrixTokens(MatType* mat, const std::vector<char>& uintToChar)
{
    if (!mat)
        return {};

    const uint32_t steps = mat->GetDimsY();
    const uint32_t cols  = mat->GetDimsX();
    std::string decoded;
    decoded.reserve(steps);

    const T* data = mat->DataRead();
    if (cols == 1u)
    {
        for (uint32_t t = 0u; t < steps; ++t)
        {
            const uint32_t idx = static_cast<uint32_t>(data[t]);
            decoded.push_back(idx < uintToChar.size() ? uintToChar[idx] : '?');
        }
        return decoded;
    }

    for (uint32_t t = 0u; t < steps; ++t)
    {
        const T* row = data + (t * cols);
        uint32_t bestIdx = 0u;
        T bestVal = row[0];
        for (uint32_t v = 1u; v < cols; ++v)
        {
            const T val = row[v];
            if (val > bestVal)
            {
                bestVal = val;
                bestIdx = v;
            }
        }
        decoded.push_back(bestIdx < uintToChar.size() ? uintToChar[bestIdx] : '?');
    }
    return decoded;
}

std::string DecodeMatrixTokensBatch(MatType* mat, const std::vector<char>& uintToChar, uint32_t batchIndex)
{
    if (!mat)
    {
        return {};
    }

    if (mat->GetDimsZ() <= 1u)
    {
        return DecodeMatrixTokens(mat, uintToChar);
    }

    MatType slice;
    mat->GetSliceZ(&slice, batchIndex);
    return DecodeMatrixTokens(&slice, uintToChar);
}


static uint32_t CountCorrectTokensBatch(MatType* outputBatchView,
                                        MatType* target,
                                        uint32_t blockSize,
                                        uint32_t vocabSize,
                                        uint32_t batchIndex)
{
    uint32_t correct = 0u;
    for (uint32_t t = 0; t < blockSize; ++t)
    {
        const uint32_t targetToken = static_cast<uint32_t>(target->GetValue(0u, t, batchIndex));
        uint32_t bestIdx = 0u;
        T bestVal = outputBatchView->GetValue(0u, t, 0u);
        for (uint32_t v = 1u; v < vocabSize; ++v)
        {
            const T val = outputBatchView->GetValue(v, t, 0u);
            if (val > bestVal)
            {
                bestVal = val;
                bestIdx = v;
            }
        }
        if (bestIdx == targetToken)
        {
            ++correct;
        }
    }
    return correct;
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

        // Init: Flatten output
        auto* finalFlatten      = new FlattenCopyLayer<T, MatType>();
        finalFlatten->SetName("ln_f_flatten");
        auto* finalFlatten_blob = InitFlattenCopyBlob<T, MatType>(true);
        model->AddLayer(finalFlatten, finalFlatten_blob);
        model->AddDependency(finalFlatten, 0, finalLayerNorm, 0);

        // Init final dense layer
        auto* finalDense      = new Dense<T, Identity<T>, MatType, true>();
        finalDense->SetName("lm_head");
        auto* finalDense_blob = InitDenseBlob<T, MatType>(vocab_size, cfg.n_embed, cfg.block_size * cfg.batch_size, true, true, cfg.use_bias, true);

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
        model->AddDependency(finalDense, 0, finalFlatten, 0);
        model->SetFinalOutputLayer(finalDense);

        const float weightDecay = 0.01f;
        auto* adamopt = new AdamWOptimiser<T, MatType>(cfg.learning_rate, 0.9f, 0.999f, 1e-8f, weightDecay);
        model->SetOptimiser(adamopt);
        model->SetLoss(new CategoricalCrossEntropyWithLogits<T, MatType>());

        model->SetExpectedInputDims({1u, cfg.block_size, cfg.batch_size});
        model->SetName(cfg.model_name);
        model->Init();
    }
    else
    {
        model->Init();
        static_cast<AdamWOptimiser<T, MatType>*>(model->GetOptimiser())->SetLearningRate(0.00003f);
        PressAnyKeyToContinue();

        // If you want to change the model after it is loaded.
        // You do it here:
        // EXAMPLE:
        //      static_cast<AdamWOptimiser<T, MatType>*>(model->GetOptimiser())->SetLearningRate(0.00001f);
        // OR:
        //      model->GetLayerBlob("embedding_dropout")->Set("Probability", 0.00f);
        // 
    }

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
    tensors.posInput = inst.AllocateMatrix({1u, cfg.block_size, 1u}, "Pos_Input");
    for (uint32_t t = 0; t < cfg.block_size; ++t)
    {
        tensors.posInput->SetValue(0u, t, 0u, static_cast<T>(t));
    }

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
    auto corpusRef = inst.AllocateMatrix({1u, static_cast<uint32_t>(uint32Data.size()), 1u}, "Corpus_Tokens");
    MatType* corpus = corpusRef.get();
    assert(corpus);
    T* corpusPtr = corpus->DataWrite();
    for (uint32_t i = 0; i < static_cast<uint32_t>(uint32Data.size()); ++i)
    {
        corpusPtr[i] = static_cast<T>(uint32Data[i]);
    }

    auto inputBufferRef  = inst.AllocateMatrix({1u, cfg.block_size, 1u}, "Input_Buffer");
    auto targetBufferRef = inst.AllocateMatrix({1u, cfg.block_size, 1u}, "Target_Buffer");
    MatType* inputBuffer = inputBufferRef.get();
    MatType* targetBuffer = targetBufferRef.get();
    assert(inputBuffer);
    assert(targetBuffer);

    // Random number for picking a start position within the corpus for training.
    const uint32_t maxStart       = static_cast<uint32_t>(uint32Data.size() - cfg.block_size - 1u);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> startDist(0u, maxStart);

    // Loop forever training.
    while(true)
    {
        for(uint32_t i = 0; i < cfg.iter_count; i++)
        {
            const uint32_t start = (maxStart == 0u ? 0u : startDist(rng));
            CopyRange(inputBuffer, corpus, 0u, start, cfg.block_size);
            CopyRange(targetBuffer, corpus, 0u, start + 1u, cfg.block_size);
            model->Run(inputBuffer, targetBuffer);

            LOG_INFO() << "Iteration: " << i << " Loss: " << model->GetLastLoss();
            // Save occaisonally
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
                    LOG_INFO() << "Last run time (s): " << lastRunTime;

                    // Dump Memory Information
                    LOG_INFO() << inst.GetString();

                    // Output a sample and accuracy for each batch.
                    float meanacc = 0.f;
                    static std::deque<float> accuracyHistory;
                    const std::string decodedIn = DecodeMatrixTokens(inputBuffer, vocab.unique);
                    LOG_INFO() << "Decoded input batch 0: \n" << decodedIn;
                    const std::string decodedTarget = DecodeMatrixTokens(targetBuffer, vocab.unique);
                    LOG_INFO() << "Decoded target batch 0: \n" << decodedTarget;
                    MatType outputBatchView;
                    output->GetSliceZ(&outputBatchView, 0u);
                    const std::string decodedOut = DecodeMatrixTokens(&outputBatchView, vocab.unique);
                    LOG_INFO() << "Decoded output batch 0: \n" << Tools::Logger::Colour(128, 255, 225) << decodedOut << Tools::Logger::ClearColour();

                    // Quick accuracy check against targets.
                    const uint32_t vocabSize = output->GetDimsX();
                    uint32_t correct = CountCorrectTokensBatch(&outputBatchView,
                                                               targetBuffer,
                                                               cfg.block_size,
                                                               vocabSize,
                                                               0u);
                    const float acc = static_cast<float>(correct) / static_cast<float>(cfg.block_size);
                    LOG_INFO() << "Batch 0 token accuracy: " << acc;
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
    /*
        Model: NanoLLM
            1  : embedding_token[Embedding]
            2  : embedding_pos[Embedding]
            3  : token_pos_add[AddLayer]
            4  : embedding_dropout[DropoutLayer]
            4+1: block_0[TransformerBlock]
            ...
            4+n: block_{n-1}[TransformerBlock]
            5+n: ln_f[LayerNorm]
            6+n: ln_f_flatten[FlattenCopy]
            7+n: lm_head[Dense]
    */
    /////////////////////////
    /* // With Dimensions: Note this codebase is ROW-MAJOR.
        Model: NanoLLM
        1 : embedding_token[Embedding]
        in : [1, block_size, batch_size]         // token IDs
        out : [n_embed, block_size, batch_size]

        2 : embedding_pos[Embedding]
        in : [1, block_size, batch_size]         // position IDs
        out : [n_embed, block_size, batch_size]

        3 : token_pos_add[AddLayer]
        in : [n_embed, block_size, batch_size] + [n_embed, block_size, batch_size]
        out : [n_embed, block_size, batch_size]

        4 : embedding_dropout[DropoutLayer]
        in : [n_embed, block_size, batch_size]
        out : [n_embed, block_size, batch_size]

        4 + 1 : block_0[TransformerBlock]
        in : [n_embed, block_size, batch_size]
        out : [n_embed, block_size, batch_size]

        ...
        4 + n : block_{ n - 1 } [TransformerBlock]
        in : [n_embed, block_size, batch_size]
        out : [n_embed, block_size, batch_size]

        5 + n : ln_f[LayerNorm]
        in : [n_embed, block_size, batch_size]
        out : [n_embed, block_size, batch_size]

        6 + n : ln_f_flatten[FlattenCopy]
        in : [n_embed, block_size, batch_size]
        out : [n_embed, block_size * batch_size, 1]

        7 + n : lm_head[Dense]
        in : [n_embed, block_size * batch_size, 1]
        out : [vocab_size, block_size * batch_size, 1]
    */

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
