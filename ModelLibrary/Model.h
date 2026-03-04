#pragma once

#include <MatrixLibrary/MatrixBase.h>
#include <MatrixLibrary/MatrixManager.h>
#include <MatrixLibrary/MatrixBase_Functions.h>

#include <OptimiserLibrary/Optimiser.h>
#include <LossLibrary/DirectLoss.h>

#include <LayerLibrary/Layer.h>
#include <LayerLibrary/Datablob.h>

#include <ToolsLibrary/Logger.h>
#include <ToolsLibrary/Timer.h>

#include <Loading/SaveLoader.h>

#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_map>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <utility>

template<typename T, class Mat = MatrixCPU<T>>
class Model
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

    public:
    struct ModelGuideLayer;
    struct ModelGuide;

    Model()
    {
    }

    ~Model()
    {
        DeleteAllLayersAndBlobs();
    }

    void DeleteAllLayersAndBlobs()
    {
        for (const auto& [layer, blob] : m_layersAndBlobs)
        {
            delete blob;
            delete layer;
        }
        m_layersAndBlobs.clear();
        m_allLayers.clear();
        m_layerDependencies.clear();

        m_timingsForward.clear();
        m_timingsBackward.clear();
        m_timingsOther.clear();

        m_rollingLossHistory.clear();
        m_rollingLossSum = 0.0;
        m_rollingLoss    = 0.0;

        m_executionOrder.clear();

        if(m_optimiser)
        {
            delete m_optimiser;
            m_optimiser = nullptr;
        }
        if (m_lossFunc)
        {
            delete m_lossFunc;
            m_lossFunc = nullptr;
        }

        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        if (m_inputMatrix.get())
        {
            inst.ReleaseSkeleton(m_inputMatrix);
        }

        if (m_outputErrorMatrix.get())
        {
            m_outputErrorMatrix = MatrixRef();
        }

    }

    bool Save()
    {
        SaveLoader<T,Mat> sl;
        std::filesystem::path path = std::filesystem::current_path() / sl.sanitize_folder_name(GetName());
        sl.EnsureEmptyDir(path);

        const std::filesystem::path oldCwd = std::filesystem::current_path();
        std::filesystem::current_path(path);

        std::vector<std::string> outfiles;
        outfiles.reserve(m_allLayers.size());
        for (auto* layer : m_allLayers)
        {
            std::string outfile = sl.SaveLayer(layer, m_layersAndBlobs[layer]);
            outfiles.push_back(outfile);
        }

        std::string lossFile;
        if (m_lossFunc)
        {
            lossFile = sl.SaveLoss(m_lossFunc);
        }

        // Save Optimiser
        std::vector<MatrixRef> trainableParams;
        {
            std::unordered_set<Mat*> seen;
            for (auto* layer : m_allLayers)
            {
                Datablob<T, Mat>* blob = m_layersAndBlobs[layer];
                std::vector<MatrixRef>* weights = layer ? layer->GetWeights(blob) : nullptr;
                if (!weights)
                {
                    continue;
                }
                for (MatrixRef& weight : *weights)
                {
                    if (!weight)
                    {
                        continue;
                    }
                    Mat* weightPtr = weight.get();
                    if (weightPtr && seen.insert(weightPtr).second)
                    {
                        trainableParams.push_back(weight);
                    }
                }
            }
        }

        std::string optFile;
        if (m_optimiser)
        {
            optFile = sl.SaveOptimiser(m_optimiser, trainableParams);
        }

        // Save dependency
        const std::filesystem::path depFile = "model.depedencylist";
        SaveDependencyList(depFile.string());

        // This will save the model outline
        // LAYER        {layerid}           {path to layer}
        // OPTIMISER    {path to optimiser} 0
        // LOSSFUNC     {path to loss}      0
        SaveModelGuide(outfiles, m_allLayers, lossFile, optFile, m_expectedInputDims);

        std::filesystem::current_path(oldCwd);
        return true;
    };

    bool Load(std::string _name)
    {
        DeleteAllLayersAndBlobs();

        SaveLoader<T,Mat> sl;
        std::filesystem::path path = std::filesystem::current_path() / sl.sanitize_folder_name(_name);

        if (!std::filesystem::exists(path))
        {
            return false;
        }

        ModelGuide mg = LoadModelGuide(path);
        for (const auto& layerEntry : mg.layers)
        {
            Datablob<T, Mat>* blob = new Datablob<T, Mat>();
            Layer<T, Mat>* layer = sl.LoadLayerFromFile(blob, layerEntry.path);
            if (!layer)
            {
                delete blob;
                continue;
            }
            AddLayer(layer, blob);
        }

        std::vector<MatrixRef> trainableParams;
        {
            std::unordered_set<Mat*> seen;
            for (auto* layer : m_allLayers)
            {
                Datablob<T, Mat>* blob = m_layersAndBlobs[layer];
                std::vector<MatrixRef>* weights = layer ? layer->GetWeights(blob) : nullptr;
                if (!weights)
                {
                    continue;
                }
                for (MatrixRef& weight : *weights)
                {
                    if (!weight)
                    {
                        continue;
                    }
                    Mat* weightPtr = weight.get();
                    if (weightPtr && seen.insert(weightPtr).second)
                    {
                        trainableParams.push_back(weight);
                    }
                }
            }
        }

        if (!mg.optimiser.empty())
        {
            m_optimiser = sl.LoadOptimiser(mg.optimiser, trainableParams);
        }
        if (!mg.loss.empty())
        {
            m_lossFunc = sl.LoadLoss(mg.loss);
        }

        if (mg.hasExpectedInputDims)
        {
            SetExpectedInputDims(mg.expectedInputDims);
        }

        LoadDependencyList((path / "model.depedencylist").string());
        SetName(_name);

        return true;
    }

    struct ModelGuideLayer
    {
        uint64_t guid = 0u;
        std::filesystem::path path;
    };

    struct ModelGuide
    {
        std::vector<ModelGuideLayer> layers;
        std::filesystem::path optimiser;
        std::filesystem::path loss;
        Dims3D expectedInputDims = Dims3D(0u, 0u, 0u);
        bool hasExpectedInputDims = false;
    };

    bool SaveDependencyList(std::string filepath) 
    {
        std::ofstream out(filepath, std::ios::out);
        if (!out)
        {
            LOG_WARNING() << "Model::SaveDependencyList: failed to open file: " << filepath;
            return false;
        }

        for (const auto& entry : m_layerDependencies)
        {
            Layer<T, Mat>* layer = entry.first;
            uint64_t layerGuid = layer ? layer->m_guid : 0u;
            for (const auto& dep : entry.second)
            {
                Layer<T, Mat>* otherLayer   = dep.first;
                const uint64_t otherGuid    = otherLayer ? otherLayer->m_guid : 0u;
                const uint32_t inputIndex   = dep.second.first;
                const uint32_t outputIndex  = dep.second.second;
                out << layerGuid << ", " << inputIndex << ", "
                    << otherGuid << ", " << outputIndex << "\n";

                LOG_INFO() << "Saving dependency: " << layer->GetName() << "(" << layer->GetID() << ") ->" << otherLayer->GetName() << "(" << otherLayer->GetID() << ").";
            }
        }

        return true;
    }

    bool LoadDependencyList(const std::string& filepath)
    {
        std::ifstream in(filepath, std::ios::in);
        if (!in)
        {
            LOG_WARNING() << "Model::LoadDependencyList: failed to open file: " << filepath;
            return false;
        }

        std::unordered_map<uint64_t, Layer<T, Mat>*> guidToLayer;
        guidToLayer.reserve(m_allLayers.size());
        for (auto* layer : m_allLayers)
        {
            if (layer)
            {
                guidToLayer[layer->m_guid] = layer;
            }
        }

        m_layerDependencies.clear();

        std::string line;
        while (std::getline(in, line))
        {
            if (line.empty())
            {
                continue;
            }

            for (char& ch : line)
            {
                if (ch == ',')
                {
                    ch = ' ';
                }
            }

            std::stringstream ss(line);
            uint64_t layerGuid = 0u;
            uint64_t otherGuid = 0u;
            uint32_t inputIndex = 0u;
            uint32_t outputIndex = 0u;
            if (!(ss >> layerGuid >> inputIndex >> otherGuid >> outputIndex))
            {
                LOG_WARNING() << "Model::LoadDependencyList: failed to parse line: " << line;
                continue;
            }

            auto layerIt = guidToLayer.find(layerGuid);
            if (layerIt == guidToLayer.end())
            {
                LOG_WARNING() << "Model::LoadDependencyList: missing layer GUID " << layerGuid;
                continue;
            }
            auto otherIt = guidToLayer.find(otherGuid);
            if (otherIt == guidToLayer.end())
            {
                LOG_WARNING() << "Model::LoadDependencyList: missing other layer GUID " << otherGuid;
                continue;
            }

            m_layerDependencies[layerIt->second].push_back({otherIt->second, {inputIndex, outputIndex}});
        }

        return true;
    }

    bool SaveModelGuide(const std::vector<std::string>& layerFiles,
                        const std::vector<Layer<T, Mat>*>& layers,
                        const std::string& lossFile,
                        const std::string& optFile,
                        const Dims3D& expectedInputDims) const
    {
        const std::string guideFile = "model.guide";
        std::ofstream out(guideFile, std::ios::out);
        if (!out)
        {
            LOG_WARNING() << "Model::SaveModelGuide: failed to open file: " << guideFile;
            return false;
        }

        const size_t count = std::min(layerFiles.size(), layers.size());
        if (layerFiles.size() != layers.size())
        {
            LOG_WARNING() << "Model::SaveModelGuide: layer file count mismatch. files="
                          << layerFiles.size() << " layers=" << layers.size();
        }

        for (size_t i = 0; i < count; ++i)
        {
            Layer<T, Mat>* layer = layers[i];
            if (!layer)
            {
                continue;
            }
            out << "LAYER " << layer->m_guid << " " << layerFiles[i] << "\n";
        }

        out << "INPUT_DIMS " << expectedInputDims.x << " "
            << expectedInputDims.y << " " << expectedInputDims.z << "\n";
        out << "OPTIMISER " << (optFile.empty() ? "0" : optFile) << " 0\n";
        out << "LOSSFUNC " << (lossFile.empty() ? "0" : lossFile) << " 0\n";
        return true;
    }

    ModelGuide LoadModelGuide(const std::filesystem::path& modelDir) const
    {
        ModelGuide guide;
        const std::filesystem::path guidePath = modelDir / "model.guide";
        std::ifstream in(guidePath, std::ios::in);
        if (!in)
        {
            LOG_WARNING() << "Model::LoadModelGuide: failed to open file: " << guidePath.string();
            return guide;
        }

        std::string line;
        while (std::getline(in, line))
        {
            if (line.empty())
            {
                continue;
            }

            std::stringstream ss(line);
            std::string tag;
            ss >> tag;
            if (tag == "LAYER")
            {
                ModelGuideLayer entry;
                std::string pathToken;
                if (!(ss >> entry.guid >> pathToken))
                {
                    LOG_WARNING() << "Model::LoadModelGuide: failed to parse LAYER line: " << line;
                    continue;
                }
                std::filesystem::path layerPath(pathToken);
                if (!layerPath.is_absolute())
                {
                    layerPath = modelDir / layerPath;
                }
                entry.path = layerPath;
                guide.layers.push_back(entry);
                continue;
            }
            if (tag == "OPTIMISER")
            {
                std::string pathToken;
                ss >> pathToken;
                if (!pathToken.empty() && pathToken != "0")
                {
                    std::filesystem::path optPath(pathToken);
                    if (!optPath.is_absolute())
                    {
                        optPath = modelDir / optPath;
                    }
                guide.optimiser = optPath;
                }
                continue;
            }
            if (tag == "LOSSFUNC")
            {
                std::string pathToken;
                ss >> pathToken;
                if (!pathToken.empty() && pathToken != "0")
                {
                    std::filesystem::path lossPath(pathToken);
                    if (!lossPath.is_absolute())
                    {
                        lossPath = modelDir / lossPath;
                    }
                    guide.loss = lossPath;
                }
                continue;
            }
            if (tag == "INPUT_DIMS")
            {
                uint32_t x = 0u;
                uint32_t y = 0u;
                uint32_t z = 0u;
                if (!(ss >> x >> y >> z))
                {
                    LOG_WARNING() << "Model::LoadModelGuide: failed to parse INPUT_DIMS line: " << line;
                    continue;
                }
                guide.expectedInputDims = Dims3D(x, y, z);
                guide.hasExpectedInputDims = true;
                continue;
            }
        }

        return guide;
    }

    void Init() 
    {
        if(m_allLayers.size() == 0)
        {
            LOG_INFO() << "Failed to Init model due to having no layers";
            assert(m_allLayers.size());
            return;
        }

        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        CalculateTotalOrdering();
        if (!m_finalOutput)
        {
            m_finalOutput = m_executionOrder.back();
        }

        // Allocate skeleton for input matrix
        if(!m_inputMatrix)
        {
            LOG_INFO() << "Allocating input skeleton matrix.";
            m_inputMatrix = inst.GetSkeleton();
            m_inputMatrix->SetDims(m_expectedInputDims);
        }

        // Link all the forward dependencies together and ensure outputs are allocated.
        for (auto* layer : m_executionOrder)
        {
            Datablob<T, Mat>* blob = m_layersAndBlobs[layer];
            auto depsIt = m_layerDependencies.find(layer);
            if (depsIt != m_layerDependencies.end())
            {
                for (auto& layerdeps : depsIt->second)
                {
                    auto*    target     = layerdeps.first;
                    uint32_t inputidx   = layerdeps.second.first;
                    uint32_t outputidx  = layerdeps.second.second;

                    MatrixRef output = target->GetOutput(m_layersAndBlobs[target], outputidx);
                    if (!output)
                    {
                        target->EnsureOutputsAllocated(m_layersAndBlobs[target]);
                        output = target->GetOutput(m_layersAndBlobs[target], outputidx);
                    }

                    if (output)
                    {
                        layer->SetInput(blob, output, inputidx);
                    }
                    else
                    {
                        LOG_WARNING() << "Missing output for layer " << target->GetName()
                                      << " when wiring input " << inputidx
                                      << " of layer " << layer->GetName();
                    }
                }
            }
            else
            {
                if (!m_inputMatrix)
                {
                    LOG_WARNING() << "Model input matrix not allocated when wiring layer " << layer->GetName();
                }
                else
                {
                    layer->SetInput(blob, m_inputMatrix, 0);
                }
            }
        }

        // With inputs now set, allocate dependent matrices
        for (auto* layer : m_executionOrder)
        {
            Datablob<T, Mat>* blob = m_layersAndBlobs[layer];
            layer->EnsureOutputsAllocated(blob);
        }

        MatrixRef finalOutputMat = m_finalOutput->GetOutput(m_layersAndBlobs[m_finalOutput], 0);
        if (!finalOutputMat)
        {
            m_finalOutput->EnsureOutputsAllocated(m_layersAndBlobs[m_finalOutput]);
            finalOutputMat = m_finalOutput->GetOutput(m_layersAndBlobs[m_finalOutput], 0);
        }

        assert(finalOutputMat);
        m_outputErrorMatrix = inst.AllocateMatrix(finalOutputMat->GetDims(), "ModelOutput_Error");
        m_layersAndBlobs[m_finalOutput]->Set(m_finalOutput->GetErrorInputName(), m_outputErrorMatrix);

        // Link all the backwards dependencies together
        for (const auto& [source, value] : m_layerDependencies)
        {
            for (auto& layerdeps : value)
            {
                auto*    target     = layerdeps.first;
                uint32_t inputidx   = layerdeps.second.first;
                uint32_t outputidx  = layerdeps.second.second;

                MatrixRef errorOut = source->GetOutputError(m_layersAndBlobs[source], inputidx);
                if (errorOut)
                {
                    target->SetErrorInput(m_layersAndBlobs[target], errorOut, outputidx);
                }
                else
                {
                    LOG_WARNING() << "Missing error output for layer " << source->GetName()
                                  << " when wiring error input " << outputidx
                                  << " of layer " << target->GetName();
                }
            }
        }
    }

    std::string GetModelInOut()
    {
        std::stringstream ss;
        if (!m_name.empty())
        {
            ss << "Model: ";
            ss << m_name;
            ss << std::endl;
        }
        for (auto* layer : m_executionOrder)
        {
            Datablob<T, Mat>* blob = m_layersAndBlobs[layer];
            ss << layer->GetInputOutputString(blob) << std::endl;
        }

        return ss.str();
    }

    void AddTiming(Layer<T,Mat>* _layer, double _time, std::map< Layer<T, Mat>* , std::vector<double> >& _storage)
    {
        if(m_enableTiming)
        {
            auto res = _storage.find(_layer);
            if(res != _storage.end())
            {
                res->second.push_back(_time);
            }
            else
            {
                auto& timings = _storage[_layer];
                timings.push_back(_time);
            }
        }
    }

    void AddTimingOther(std::string _name, double _time, std::map< std::string, std::vector<double> >& _storage)
    {
        if (m_enableTiming)
        {
            auto res = _storage.find(_name);
            if (res != _storage.end())
            {
                res->second.push_back(_time);
            }
            else
            {
                auto& timings = _storage[_name];
                timings.push_back(_time);
            }
        }
    }

    void PrintTimings()
    {
        if (!m_enableTiming)
        {
            LOG_INFO() << "Timing is disabled.";
            return;
        }

        size_t maxNameLen = 5;
        for (auto* layer : m_executionOrder)
        {
            const size_t len = layer->GetName().size();
            if (len > maxNameLen)
            {
                maxNameLen = len;
            }
        }

        LOG_INFO() << "Timing summary for model: " << m_name;
        LOG_INFOF("%-*s  %12s %12s %6s  %12s %12s %6s",
                  static_cast<int>(maxNameLen), "Layer",
                  "FwdAvg(ms)", "FwdLast(ms)", "FwdN",
                  "BwdAvg(ms)", "BwdLast(ms)", "BwdN");

        for (auto* layer : m_executionOrder)
        {
            double fwdAvg = 0.0, fwdLast = 0.0;
            size_t fwdCount = 0;
            auto fit = m_timingsForward.find(layer);
            if (fit != m_timingsForward.end() && !fit->second.empty())
            {
                const auto& v = fit->second;
                fwdCount = v.size();
                double sum = 0.0;
                for (double t : v)
                {
                    sum += t;
                }
                fwdAvg = sum / static_cast<double>(fwdCount);
                fwdLast = v.back();
            }

            double bwdAvg = 0.0, bwdLast = 0.0;
            size_t bwdCount = 0;
            auto bit = m_timingsBackward.find(layer);
            if (bit != m_timingsBackward.end() && !bit->second.empty())
            {
                const auto& v = bit->second;
                bwdCount = v.size();
                double sum = 0.0;
                for (double t : v)
                {
                    sum += t;
                }
                bwdAvg = sum / static_cast<double>(bwdCount);
                bwdLast = v.back();
            }

            LOG_INFOF("%-*s  %12.3f %12.3f %6zu  %12.3f %12.3f %6zu",
                      static_cast<int>(maxNameLen), layer->GetName().c_str(),
                      fwdAvg * 1000.0, fwdLast * 1000.0, fwdCount,
                      bwdAvg * 1000.0, bwdLast * 1000.0, bwdCount);
        }

        if (!m_timingsOther.empty())
        {
            size_t otherNameLen = 5;
            for (const auto& item : m_timingsOther)
            {
                const size_t len = item.first.size();
                if (len > otherNameLen)
                {
                    otherNameLen = len;
                }
            }

            LOG_INFO() << "Other timing metrics:";
            LOG_INFOF("%-*s  %12s %12s %6s",
                      static_cast<int>(otherNameLen), "Name",
                      "Avg(ms)", "Last(ms)", "N");

            for (const auto& item : m_timingsOther)
            {
                const auto& v = item.second;
                double avg = 0.0;
                double last = 0.0;
                size_t count = v.size();
                if (count > 0)
                {
                    double sum = 0.0;
                    for (double t : v)
                    {
                        sum += t;
                    }
                    avg = sum / static_cast<double>(count);
                    last = v.back();
                }

                LOG_INFOF("%-*s  %12.3f %12.3f %6zu",
                          static_cast<int>(otherNameLen), item.first.c_str(),
                          avg * 1000.0, last * 1000.0, count);
            }
        }
    }

    std::string GetString()
    {
        std::string out;
        out.reserve(m_executionOrder.size() * 64u + 32u);
        if (!m_name.empty())
        {
            out.append("Model: ");
            out.append(m_name);
            out.push_back('\n');
        }

        const size_t count = m_executionOrder.size();
        for (size_t i = 0; i < count; ++i)
        {
            Layer<T, Mat>* layer = m_executionOrder[i];
            out.append(std::to_string(i + 1u));
            out.append(": ");
            out.append(layer->GetName());
            out.append(" [");
            out.append(layer->GetTypeName());
            out.append("]\n");
        }

        return out;
    }

    void Run(Mat* _input, Mat* _target, bool storeLoss = false)
    {
        if (!m_lossFunc)
        {
            m_lossFunc = new DirectLoss<T,Mat>();
        }

		Timer totalTimer;
		totalTimer.Start();
        if (_input && m_inputMatrix)
        {
            m_inputMatrix.get()->SetData(_input->GetData(), _input->GetOffset());
            m_inputMatrix.get()->SetDims(_input->GetDims());
        }

        // Do forward pass.
        for(auto* layer : m_executionOrder)
        {
            Timer forwardTimer;
            forwardTimer.Start();
            //LOG_INFO() << layer->GetString(m_layersAndBlobs[layer]) << ": Forwards";
			layer->Forward(m_layersAndBlobs[layer]);
            double elapsed = forwardTimer.Elapsed();
            AddTiming(layer, elapsed, m_timingsForward);
        }

        if(_target)
        {
            // Calculate Error
            Mat* finalOutputMat = m_layersAndBlobs[m_finalOutput]->GetMatrix(m_finalOutput->GetOutputName());

            Timer lossTimer;
            lossTimer.Start();
            m_lossFunc->Gradient(m_outputErrorMatrix.get(), _target, finalOutputMat);
            double loss = 0.0f;
            if(storeLoss)
            {
                loss = m_lossFunc->Loss(_target, finalOutputMat);

                // Rolling window average
                if (m_rollingLossHistory.size() >= m_rollingLossWindow)
                {
                    m_rollingLossSum -= m_rollingLossHistory.front();
                    m_rollingLossHistory.pop_front();
                }

                m_rollingLossHistory.push_back(loss);
                m_rollingLossSum += loss;
                m_rollingLoss = m_rollingLossSum / static_cast<double>(m_rollingLossHistory.size());
                m_lastLoss = loss;
            }



            double losselapsed = lossTimer.Elapsed();
            AddTimingOther("Loss Func", losselapsed, m_timingsOther);

            // Do backwards pass.
            for (auto it = m_executionOrder.rbegin(); it != m_executionOrder.rend(); ++it)
            {
                Layer<T, Mat>* layer = *it;
                //LOG_INFO() << "Running backwards layer: " << layer->GetName();
                Timer forwardTimer;
                forwardTimer.Start();
                //LOG_INFO() << layer->GetString(m_layersAndBlobs[layer]) << ": Backwards";
                layer->Backwards(m_layersAndBlobs[layer]);
                double elapsed = forwardTimer.Elapsed();
                AddTiming(layer, elapsed, m_timingsBackward);
            }

            if(m_optimiser)
            {
                Timer optTimer;
                optTimer.Start();
                // Update weights with gradients, aggregating tied parameters.
                std::unordered_map<Mat*, Mat*> gradAccum;
                for (auto* layer : m_executionOrder)
                {
                    Datablob<T, Mat>* data = m_layersAndBlobs[layer];
                    std::vector<MatrixRef>* trainableParams = layer->GetWeights(data);
                    std::vector<MatrixRef>* grads = layer->GetGradients(data);

                    assert(trainableParams->size() == grads->size());

                    for (size_t i = 0; i < grads->size(); ++i)
                    {
                        Mat* grad = (*grads)[i].get();
                        Mat* trainable = (*trainableParams)[i].get();
                        assert(grad);
                        assert(trainable);

                        auto it = gradAccum.find(trainable);
                        if (it == gradAccum.end())
                        {
                            gradAccum.emplace(trainable, grad);
                        }
                        else
                        {
                            Add(it->second, it->second, grad);
                        }
                    }
                }

                for (const auto& entry : gradAccum)
                {
                    m_optimiser->Step(entry.first, entry.second);
                }
                double elapsed = optTimer.Elapsed();
                AddTimingOther(m_optimiser->GetName(), elapsed, m_timingsOther);
            }
            else
            {
                LOG_WARNING() << "No weights updated because no Optimiser has been set.";
            }
        }

        double totalElapsed = totalTimer.Elapsed();
        AddTimingOther("Total Run", totalElapsed, m_timingsOther);
    }

    void AddLayer(Layer<T,Mat>* _layer, Datablob<T, Mat>* _blob)
    {
        m_allLayers.push_back(_layer);
        m_layersAndBlobs[_layer] = _blob;

        m_executionOrder.push_back(_layer);
    }

    void AddDependency(Layer<T,Mat>* _layer, uint32_t _inputIdx, Layer<T,Mat>* _otherlayer,  uint32_t _outputIdx )
    {
        if(m_layerDependencies.find(_layer) == m_layerDependencies.end())
        {
            m_layerDependencies[_layer] = {};
        }
        m_layerDependencies[_layer].push_back({_otherlayer, {_inputIdx, _outputIdx, }});
    }

    void SetFinalOutputLayer(Layer<T,Mat>* _layer)
    {
        m_finalOutput = _layer;
	}

    void CalculateTotalOrdering()
    {
        std::unordered_map<Layer<T, Mat>*, std::vector<Layer<T, Mat>*>> adjacency;
        std::unordered_map<Layer<T, Mat>*, uint32_t> indegree;
        indegree.reserve(m_allLayers.size());
        adjacency.reserve(m_allLayers.size());

        for (auto* layer : m_allLayers)
        {
            indegree[layer] = 0;
        }

        for (const auto& [source, deps] : m_layerDependencies)
        {
            for (const auto& dep : deps)
            {
                auto* target = dep.first;
                adjacency[target].push_back(source);
                ++indegree[source];
            }
        }

        std::vector<Layer<T, Mat>*> ordered;
        ordered.reserve(m_allLayers.size());
        std::vector<Layer<T, Mat>*> queue;
        queue.reserve(m_allLayers.size());

        for (auto* layer : m_allLayers)
        {
            if (indegree[layer] == 0)
            {
                queue.push_back(layer);
            }
        }

        size_t head = 0;
        while (head < queue.size())
        {
            Layer<T, Mat>* layer = queue[head++];
            ordered.push_back(layer);

            auto it = adjacency.find(layer);
            if (it == adjacency.end())
            {
                continue;
            }

            for (auto* next : it->second)
            {
                uint32_t& deg = indegree[next];
                if (--deg == 0)
                {
                    queue.push_back(next);
                }
            }
        }

        if (ordered.size() != m_allLayers.size())
        {
            LOG_INFO() << "Cycle detected in model layer dependencies.";
            for (auto* layer : m_allLayers)
            {
                if (indegree[layer] != 0)
                {
                    ordered.push_back(layer);
                }
            }
        }

        m_executionOrder.swap(ordered);
    }

    Mat* GetOutput(uint32_t _index = 0u)
    {
        if(m_finalOutput)
        {
            MatrixRef output = m_finalOutput->GetOutput(m_layersAndBlobs[m_finalOutput], _index);
            return output.get();
        }
        else
            return nullptr;
    }

    Mat* GetFinalError()
    {
        return m_outputErrorMatrix.get();
    }

    T GetLastLoss()
    {
        return m_lastLoss;
    }

    double GetRollingLoss()
    {
        return m_rollingLoss;
    }

    void SetRollingLossWindow(uint32_t _window)
    {
        m_rollingLossWindow = (_window > 0u) ? _window : 1u;
        while (m_rollingLossHistory.size() > m_rollingLossWindow)
        {
            m_rollingLossSum -= m_rollingLossHistory.front();
            m_rollingLossHistory.pop_front();
        }
        if (!m_rollingLossHistory.empty())
            m_rollingLoss = m_rollingLossSum / static_cast<double>(m_rollingLossHistory.size());
    }

    void SetLoss(LossBase<T,Mat>* _in)
    {
        m_lossFunc = _in;
    }
    
    Optimiser<T, Mat>* GetOptimiser()
    {
        return m_optimiser;
    }

    void SetOptimiser( Optimiser<T,Mat>* _opt)
    {
        m_optimiser = _opt;
    }

    Layer<T,Mat>* GetFinalOutputLayer()
    {
        return m_finalOutput;
	}

    Layer<T, Mat>* GetLayerByName(const std::string& _name)
    {
        for(auto* layer : m_allLayers)
        {
            if(layer->GetName() == _name)
                return layer;
        }
        return nullptr;
	}

    Layer<T, Mat>* GetLayer(const std::string& _name)
    {
        return GetLayerByName(_name);
    }

    Datablob<T, Mat>* GetLayerBlob(const std::string& _name)
    {
        Layer<T, Mat>* layer = GetLayerByName(_name);
        if (!layer)
        {
            return nullptr;
        }
        auto it = m_layersAndBlobs.find(layer);
        if (it == m_layersAndBlobs.end())
        {
            return nullptr;
        }
        return it->second;
    }

    void SetExpectedInputDims(const Dims3D& _dims)
    {
        m_expectedInputDims = _dims;
	}

    Dims3D GetExpectedInputDims() const
    {
        return m_expectedInputDims;
    }

    void SetName(std::string _name)
    {
        m_name = _name;
    }

    std::string GetName()
    {
        return m_name;
    }

    double GetLastRunTime_Seconds()
    {
        double total = 0.0;
        auto it = m_timingsOther.find("Total Run");
        if (it != m_timingsOther.end() && !it->second.empty())
        {
            total = it->second.back();
        }
        return total;
	}

    MatrixRef   m_inputMatrix;
	Dims3D      m_expectedInputDims         = Dims3D(0,0);
    MatrixRef   m_outputErrorMatrix;
    T           m_lastLoss                  = T(0);

    uint32_t           m_rollingLossWindow  = 100u;
    std::deque<double> m_rollingLossHistory;
    double             m_rollingLossSum     = 0.0;
    double             m_rollingLoss        = 0.0;

    LossBase<T, Mat>*  m_lossFunc    = nullptr;
    Optimiser<T, Mat>* m_optimiser   = nullptr;
    Layer<T, Mat>*     m_finalOutput = nullptr;

    std::vector<Layer<T, Mat>*>                         m_executionOrder;
    std::vector  <Layer<T, Mat>*>                       m_allLayers;
    std::map     <Layer<T, Mat>*, Datablob<T, Mat>*>    m_layersAndBlobs;
    std::map     <Layer<T,Mat>*,  std::vector< std::pair<Layer<T,Mat>*, std::pair<uint32_t, uint32_t>> > > m_layerDependencies;

    bool                                                 m_enableTiming = true;
    std::map     <Layer<T, Mat>*, std::vector<double>>   m_timingsForward;
    std::map     <Layer<T, Mat>*, std::vector<double>>   m_timingsBackward;
    std::map     <std::string, std::vector<double>>      m_timingsOther;
    

    std::string m_name = "No Name Set";

    private:
    void EnsureInputMatrix(const Dims3D& dims)
    {
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        if (m_inputMatrix)
        {
            if (m_inputMatrix->GetDimsX() == dims.x
                && m_inputMatrix->GetDimsY() == dims.y
                && m_inputMatrix->GetDimsZ() == dims.z)
            {
                return;
            }
            inst.RemoveMatrix(m_inputMatrix);
        }
        m_inputMatrix = inst.AllocateMatrix(dims, "ModelInput");
    }
};
