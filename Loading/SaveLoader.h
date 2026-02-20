
#pragma once 
#include <LayerLibrary/Layer.h>
#include <LayerLibrary/Datablob.h>
#include <LayerLibrary/ActivationLayer.h>
#include <LayerLibrary/AddLayer.h>
#include <LayerLibrary/CausalSelfAttentionLayer.h>
#include <LayerLibrary/Conv2D.h>
#include <LayerLibrary/Dense.h>
#include <LayerLibrary/DropoutLayer.h>
#include <LayerLibrary/Embedding.h>
#include <LayerLibrary/FlattenCopyLayer.h>
#include <LayerLibrary/LayerNorm.h>
#include <LayerLibrary/TransformerBlock.h>
#include <ActivationLibrary/Activations.h>
#include <MatrixLibrary/MatrixManager.h>
#include <LossLibrary/Losses.h>
#include <OptimiserLibrary/AdamOptimiser.h>
#include <OptimiserLibrary/AdamWOptimiser.h>
#include <OptimiserLibrary/BasicOptimiser.h>
#include <ToolsLibrary/Tools.h>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <sstream>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <unordered_set>

template<class T, class Mat>
class SaveLoader
{
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;

    struct OptimiserParamFile;
    struct OptimiserFile;
    struct LossFile;

    public:
    SaveLoader(){};

    std::string sanitize_folder_name(std::string name) 
    {
        const std::string invalid = "<>:\"/\\|?*";

        for (char& c : name) {
            if (invalid.find(c) != std::string::npos || std::iscntrl(static_cast<unsigned char>(c))) {
                c = '_';
            }
        }

        // Trim trailing dots and spaces (Windows)
        while (!name.empty() && (name.back() == ' ' || name.back() == '.')) {
            name.pop_back();
        }

        if (name.empty()) {
            name = "_";
        }

        return name;
    }

    void EnsureEmptyDir(const std::filesystem::path& dir)
    {
        if (std::filesystem::exists(dir))
        {
            std::filesystem::remove_all(dir); // delete contents and the dir itself
        }
        std::filesystem::create_directories(dir);
    }

    Layer<T,Mat>* Load(Datablob<T,Mat>* _blobout, std::string _name, std::string _outdir)
    {
        // Find all *.blob_layer files in _outDir
        std::filesystem::path outDirPath(_outdir);
        if (!std::filesystem::exists(outDirPath))
        {
            LOG_WARNING() << "SaveLoader::Load: output dir does not exist: " << _outdir;
            return nullptr;
        }

        std::vector<LayerFile> layerFiles;
        for (const auto& entry : std::filesystem::directory_iterator(outDirPath))
        {
            if (!entry.is_regular_file() || entry.path().extension() != ".blob_layer")
            {
                continue;
            }

            LayerFile layerFile = ParseLayerFile(entry.path());
            if (!layerFile.path.empty())
            {
                layerFiles.push_back(std::move(layerFile));
            }
            else
            {
                LOG_WARNING() << "SaveLoader::Load: failed to parse layer file: " << entry.path().string();
            }
        }

        // Load each into a local struct

        // Find which one's name matches _name.
        LayerFile* selectedFile = nullptr;
        for (auto& lf : layerFiles)
        {
            if (lf.rootLayerName == _name)
            {
                selectedFile = &lf;
                break;
            }
        }

        if (!selectedFile)
        {
            LOG_WARNING() << "SaveLoader::Load: no matching layer named '" << _name
                          << "' found in dir: " << _outdir;
            return nullptr;
        }

        // Init a new Layer based on the ROOTTYPE in the Layer_Name entry.
        Layer<T, Mat>* rootLayer = CreateLayerFromRootType(*selectedFile);
        if (!rootLayer)
        {
            LOG_WARNING() << "SaveLoader::Load: unsupported root layer type: " << selectedFile->rootLayerType;
            return nullptr;
        }

        if (!selectedFile->rootLayerName.empty())
        {
            rootLayer->SetName(selectedFile->rootLayerName);
        }
        if (selectedFile->hasRootLayerGuid)
        {
            rootLayer->m_guid = selectedFile->rootLayerGuid;
        }

        // Populate _blobout with the data within, loading data from DATA tag in each entry, dont handle LAYER_ entries yet.
        PopulateBlobFromEntries(_blobout, *selectedFile);

        // For each LAYER_* type in the file, load its sub data into a blob and add it and its layer to _blobout.
        PopulateLayerEntries(_blobout, *selectedFile);
        
        return rootLayer;
    }

    Layer<T, Mat>* LoadLayerFromFile(Datablob<T, Mat>* _blobout, const std::filesystem::path& layerFilePath)
    {
        LayerFile file = ParseLayerFile(layerFilePath);
        if (file.path.empty())
        {
            LOG_WARNING() << "SaveLoader::LoadLayerFromFile: failed to parse layer file: "
                          << layerFilePath.string();
            return nullptr;
        }

        Layer<T, Mat>* rootLayer = CreateLayerFromRootType(file);
        if (!rootLayer)
        {
            LOG_WARNING() << "SaveLoader::LoadLayerFromFile: unsupported root layer type: "
                          << file.rootLayerType;
            return nullptr;
        }

        if (!file.rootLayerName.empty())
        {
            rootLayer->SetName(file.rootLayerName);
        }
        if (file.hasRootLayerGuid)
        {
            rootLayer->m_guid = file.rootLayerGuid;
        }

        LOG_INFO() << "Loading Layer: " << rootLayer->GetID() << ". " << rootLayer->GetName();

        PopulateBlobFromEntries(_blobout, file);
        PopulateLayerEntries(_blobout, file);
        return rootLayer;
    }

    void Save(Layer<T, Mat>* _layer, Datablob<T,Mat>* _blob, std::string outDir)
    {
        const std::filesystem::path old = std::filesystem::current_path();
        EnsureEmptyDir(old / outDir);
        std::filesystem::current_path(old / outDir);

        SaveLayer(_layer, _blob);

        std::filesystem::current_path(old);
    }

    std::string SaveOptimiser(Optimiser<T, Mat>* opt, const std::vector<MatrixRef>& params)
    {
        if (!opt)
        {
            LOG_WARNING() << "SaveLoader::SaveOptimiser: optimiser is null.";
            return {};
        }

        std::unordered_set<Mat*> paramSet;
        if (!params.empty())
        {
            paramSet.reserve(params.size());
            for (const MatrixRef& param : params)
            {
                Mat* paramPtr = param.get();
                if (paramPtr)
                {
                    paramSet.insert(paramPtr);
                }
            }
        }

        std::stringstream ss;
        const std::string type = opt->GetName();
        ss << "TYPE " << type << "\n";

        if (auto* adamw = dynamic_cast<AdamWOptimiser<T, Mat>*>(opt))
        {
            ss << "LR " << adamw->GetLearningRate() << "\n";
            ss << "BETA1 " << adamw->GetBeta1() << "\n";
            ss << "BETA2 " << adamw->GetBeta2() << "\n";
            ss << "EPS " << adamw->GetEpsilon() << "\n";
            ss << "WEIGHT_DECAY " << adamw->GetWeightDecay() << "\n";
            ss << ";\n";

            std::vector<typename AdamWOptimiser<T, Mat>::ParamState> state;
            adamw->ExportState(state);
            for (const auto& entry : state)
            {
                if (!entry.param || !entry.moment0 || !entry.moment1)
                {
                    continue;
                }
                if (!paramSet.empty() && paramSet.find(entry.param.get()) == paramSet.end())
                {
                    continue;
                }
                ss << "PARAM_GUID " << entry.param->m_guid << "\n";
                ss << "M0 " << WriteMatrixFile(entry.moment0.get()) << "\n";
                ss << "M1 " << WriteMatrixFile(entry.moment1.get()) << "\n";
                ss << "T " << entry.timestep << "\n";
                ss << ";\n";
            }
            return WriteOptimiserFile(ss.str());
        }

        if (auto* adam = dynamic_cast<AdamOptimiser<T, Mat>*>(opt))
        {
            ss << "LR " << adam->GetLearningRate() << "\n";
            ss << "BETA1 " << adam->GetBeta1() << "\n";
            ss << "BETA2 " << adam->GetBeta2() << "\n";
            ss << "EPS " << adam->GetEpsilon() << "\n";
            ss << ";\n";

            std::vector<typename AdamOptimiser<T, Mat>::ParamState> state;
            adam->ExportState(state);
            for (const auto& entry : state)
            {
                if (!entry.param || !entry.moment0 || !entry.moment1)
                {
                    continue;
                }
                if (!paramSet.empty() && paramSet.find(entry.param.get()) == paramSet.end())
                {
                    continue;
                }
                ss << "PARAM_GUID " << entry.param->m_guid << "\n";
                ss << "M0 " << WriteMatrixFile(entry.moment0.get()) << "\n";
                ss << "M1 " << WriteMatrixFile(entry.moment1.get()) << "\n";
                ss << "T " << entry.timestep << "\n";
                ss << ";\n";
            }
            return WriteOptimiserFile(ss.str());
        }

        if (auto* basic = dynamic_cast<BasicOptimiser<T, Mat>*>(opt))
        {
            ss << "LR " << basic->GetLearningRate() << "\n";
            ss << ";\n";
            return WriteOptimiserFile(ss.str());
        }

        LOG_WARNING() << "SaveLoader::SaveOptimiser: unsupported optimiser type: " << type;
        return {};
    }

    Optimiser<T, Mat>* LoadOptimiser(const std::filesystem::path& optFile,
                                     const std::vector<MatrixRef>& params)
    {
        OptimiserFile file;
        if (!ParseOptimiserFile(optFile, file))
        {
            return nullptr;
        }

        Optimiser<T, Mat>* optimiser = nullptr;
        const std::string type = file.type;
        if (type == "AdamWOptimiser")
        {
            const double lr = file.hasLr ? file.lr : 0.001;
            const double beta1 = file.hasBeta1 ? file.beta1 : 0.9;
            const double beta2 = file.hasBeta2 ? file.beta2 : 0.999;
            const double eps = file.hasEps ? file.eps : 1e-8;
            const double weightDecay = file.hasWeightDecay ? file.weightDecay : 0.01;
            optimiser = new AdamWOptimiser<T, Mat>(static_cast<T>(lr),
                                                   static_cast<T>(beta1),
                                                   static_cast<T>(beta2),
                                                   static_cast<T>(eps),
                                                   static_cast<T>(weightDecay));
        }
        else if (type == "AdamOptimiser")
        {
            const double lr = file.hasLr ? file.lr : 0.001;
            const double beta1 = file.hasBeta1 ? file.beta1 : 0.9;
            const double beta2 = file.hasBeta2 ? file.beta2 : 0.999;
            const double eps = file.hasEps ? file.eps : 1e-8;
            optimiser = new AdamOptimiser<T, Mat>(static_cast<T>(lr),
                                                  static_cast<T>(beta1),
                                                  static_cast<T>(beta2),
                                                  static_cast<T>(eps));
        }
        else if (type == "BasicOptimiser")
        {
            const double lr = file.hasLr ? file.lr : 0.001;
            optimiser = new BasicOptimiser<T, Mat>(static_cast<T>(lr));
        }
        else
        {
            LOG_WARNING() << "SaveLoader::LoadOptimiser: unsupported optimiser type: " << type;
            return nullptr;
        }

        if (file.params.empty())
        {
            return optimiser;
        }

        std::unordered_map<uint64_t, MatrixRef> paramsByGuid;
        paramsByGuid.reserve(params.size());
        for (const MatrixRef& param : params)
        {
            if (param)
            {
                paramsByGuid[param->m_guid] = param;
            }
        }

        const std::filesystem::path baseDir = optFile.parent_path();
        if (auto* adamw = dynamic_cast<AdamWOptimiser<T, Mat>*>(optimiser))
        {
            std::vector<typename AdamWOptimiser<T, Mat>::ParamState> state;
            state.reserve(file.params.size());
            for (const auto& entry : file.params)
            {
                if (!entry.hasGuid)
                {
                    LOG_WARNING() << "SaveLoader::LoadOptimiser: param entry missing GUID.";
                    continue;
                }
                auto it = paramsByGuid.find(entry.paramGuid);
                if (it == paramsByGuid.end())
                {
                    LOG_WARNING() << "SaveLoader::LoadOptimiser: missing param GUID "
                                  << entry.paramGuid << " in provided params.";
                    continue;
                }

                if (!entry.hasM0 || !entry.hasM1)
                {
                    LOG_WARNING() << "SaveLoader::LoadOptimiser: missing moment file entries for GUID "
                                  << entry.paramGuid;
                    continue;
                }
                MatrixRef m0 = LoadMatrixFromFile(baseDir / entry.m0File);
                MatrixRef m1 = LoadMatrixFromFile(baseDir / entry.m1File);
                if (!m0 || !m1)
                {
                    LOG_WARNING() << "SaveLoader::LoadOptimiser: failed to load moment matrices for GUID "
                                  << entry.paramGuid;
                    continue;
                }

                typename AdamWOptimiser<T, Mat>::ParamState ps;
                ps.param    = it->second;
                ps.moment0  = m0;
                ps.moment1  = m1;
                ps.timestep = entry.timestep;
                state.push_back(ps);
            }
            adamw->ImportState(paramsByGuid, state);
        }
        else if (auto* adam = dynamic_cast<AdamOptimiser<T, Mat>*>(optimiser))
        {
            std::vector<typename AdamOptimiser<T, Mat>::ParamState> state;
            state.reserve(file.params.size());
            for (const auto& entry : file.params)
            {
                if (!entry.hasGuid)
                {
                    LOG_WARNING() << "SaveLoader::LoadOptimiser: param entry missing GUID.";
                    continue;
                }
                auto it = paramsByGuid.find(entry.paramGuid);
                if (it == paramsByGuid.end())
                {
                    LOG_WARNING() << "SaveLoader::LoadOptimiser: missing param GUID "
                                  << entry.paramGuid << " in provided params.";
                    continue;
                }

                if (!entry.hasM0 || !entry.hasM1)
                {
                    LOG_WARNING() << "SaveLoader::LoadOptimiser: missing moment file entries for GUID "
                                  << entry.paramGuid;
                    continue;
                }
                MatrixRef m0 = LoadMatrixFromFile(baseDir / entry.m0File);
                MatrixRef m1 = LoadMatrixFromFile(baseDir / entry.m1File);
                if (!m0 || !m1)
                {
                    LOG_WARNING() << "SaveLoader::LoadOptimiser: failed to load moment matrices for GUID "
                                  << entry.paramGuid;
                    continue;
                }

                typename AdamOptimiser<T, Mat>::ParamState ps;
                ps.param = it->second;
                ps.moment0 = m0;
                ps.moment1 = m1;
                ps.timestep = entry.timestep;
                state.push_back(ps);
            }
            adam->ImportState(paramsByGuid, state);
        }

        return optimiser;
    }

    std::string SaveLoss(LossBase<T, Mat>* loss)
    {
        if (!loss)
        {
            LOG_WARNING() << "SaveLoader::SaveLoss: loss is null.";
            return {};
        }

        std::stringstream ss;
        ss << "TYPE " << loss->GetName() << "\n";
        ss << ";\n";
        return WriteLossFile(ss.str());
    }

    LossBase<T, Mat>* LoadLoss(const std::filesystem::path& lossFile)
    {
        LossFile file;
        if (!ParseLossFile(lossFile, file))
        {
            return nullptr;
        }

        const std::string type = file.type;
        if (type == "DirectLoss")
        {
            return new DirectLoss<T, Mat>();
        }
        if (type == "DirectNormalisedLoss")
        {
            return new DirectNormalisedLoss<T, Mat>();
        }
        if (type == "MeanSquaredError")
        {
            return new MeanSquaredError<T, Mat>();
        }
        if (type == "BinaryCrossEntropy")
        {
            return new BinaryCrossEntropy<T, Mat>();
        }
        if (type == "CategoricalCrossEntropyWithLogits")
        {
            return new CategoricalCrossEntropyWithLogits<T, Mat>();
        }

        LOG_WARNING() << "SaveLoader::LoadLoss: unsupported loss type: " << type;
        return nullptr;
    }

    std::string SaveLayer(Layer<T, Mat>* _layer, Datablob<T,Mat>* _blob)
    {
        std::vector<typename Layer<T,Mat>::sublayerinfo> subLayers;
        _layer->GetSublayerPairs(subLayers, _blob);

        // Dump all sub blobs

        // Write Layer
        const auto& matdata = _blob->GetAllMatrixRefData();
        const std::map<std::string, int>&       intdata      = _blob->GetAllIntData();
        const std::map<std::string, uint32_t>&  uintdata     = _blob->GetAllUIntData();
        const std::map<std::string, float>&     floatdata    = _blob->GetAllFloatData();

        std::stringstream ss;

        LOG_INFO() << "Writing Layer: " << _layer->GetID() << ". " << _layer->GetName();
        // LayerName
        ss << "NAME " << "Layer_Name" << "\n";
        ss << "DATA " << _layer->GetName() << "\n";
        ss << "TYPE " << "ROOTLAYER_" << _layer->GetTypeName() << "\n";
        ss << "META " << _layer->GetMetaData() << "\n";
        ss << "GUID " << _layer->m_guid << "\n";
        ss << ";\n";

        for( auto& _p : subLayers)
        {
            std::string bdumpfile = SaveLayer(_p.layer, _p.data);
            ss << "NAME " << _p.name << "\n";
            ss << "DATA " << bdumpfile << "\n";
            ss << "TYPE " << "LAYER_" << _p.layer->GetTypeName() << "\n";
            ss << ";\n";
        }

        // int
        for (const auto& [key, value] : intdata) 
        {
            std::string binaryDataFile = WriteBinaryDataFile(&value, sizeof(int));
            ss << "NAME " << key << "\n";
            ss << "DATA " << binaryDataFile << "\n";
            ss << "TYPE " << "INT" << "\n";
            ss << ";\n";
        }

        // uint
        for (const auto& [key, value] : uintdata) 
        {
            std::string binaryDataFile = WriteBinaryDataFile(&value, sizeof(unsigned int));
            ss << "NAME " << key << "\n";
            ss << "DATA " << binaryDataFile << "\n";
            ss << "TYPE " << "UINT" << "\n";
            ss << ";\n";
        }

        // float
        for (const auto& [key, value] : floatdata) 
        {
            std::string binaryDataFile = WriteBinaryDataFile(&value, sizeof(float));
            ss << "NAME " << key << "\n";
            ss << "DATA " << binaryDataFile << "\n";
            ss << "TYPE " << "FLOAT" << "\n";
            ss << ";\n";
        }

        // Matrix
        for (const auto& [key, value] : matdata) 
        {
            Mat* mat = value.get();
            if (!mat)
            {
                continue;
            }
            if (MatrixManager<T, Mat>::Instance().IsSkeleton(mat))
            {
                continue;
            }
            std::string matrixDataFile = WriteMatrixFile(mat);
            ss << "NAME " << key << "\n";
            ss << "DATA " << matrixDataFile << "\n";
            ss << "TYPE " << "MATRIX" << "\n";
            ss << ";\n";
            LOG_INFO() << "Wrote: " << mat->GetID() << ":" << mat->GetName() << ": " << mat->GetDims().GetString();
        }

        std::string writtenfile = WriteBlobFile(ss.str());
        LOG_INFO() << "Wrote " << writtenfile;
        return writtenfile;
        // NAME
        // NAME_Data
        // NAME_Type

        // NAME output_0
        // NAME_Data 12345.blob_matrix
        // NAME_Type Matrix

        // NAME layer_0
        // NAME_Data 12345.blob_layer
        // NAME_Type Layer

        // NAME int_0/uint_0/float_0
        // Name_Data 12345.blob_int/uint/float
        // NAME_Type INT/UINT/FLOAT

        //// blob_int/blob_uint/blob_float (binary)

        //// blob_matrix
        //// internal_name "string"
        //// dimx          2
        //// dimy          3
        //// dimz          1
        //// data          "12345.blob_data"
    }
    
    private:
    struct LayerFileEntry
    {
        std::string name;
        std::string data;
        std::string type;
        std::string meta;
        uint64_t guid = 0u;
        bool hasGuid = false;
    };

    struct LayerFile
    {
        std::filesystem::path path;
        std::vector<LayerFileEntry> entries;
        std::string rootLayerName;
        std::string rootLayerType;
        std::string rootLayerMeta;
        uint64_t rootLayerGuid = 0u;
        bool hasRootLayerGuid = false;
    };

    struct OptimiserParamFile
    {
        uint64_t paramGuid = 0u;
        std::string m0File;
        std::string m1File;
        uint64_t timestep = 0u;
        bool hasGuid = false;
        bool hasM0 = false;
        bool hasM1 = false;
        bool hasTimestep = false;
    };

    struct OptimiserFile
    {
        std::string type;
        double lr = 0.0;
        double beta1 = 0.0;
        double beta2 = 0.0;
        double eps = 0.0;
        double weightDecay = 0.0;
        bool hasLr = false;
        bool hasBeta1 = false;
        bool hasBeta2 = false;
        bool hasEps = false;
        bool hasWeightDecay = false;
        std::vector<OptimiserParamFile> params;
    };

    struct LossFile
    {
        std::string type;
    };

    static void Trim(std::string& s)
    {
        const char* whitespace = " \t\r\n";
        const size_t begin = s.find_first_not_of(whitespace);
        if (begin == std::string::npos)
        {
            s.clear();
            return;
        }
        const size_t end = s.find_last_not_of(whitespace);
        s = s.substr(begin, end - begin + 1);
    }

    static LayerFile ParseLayerFile(const std::filesystem::path& path)
    {
        std::ifstream in(path);
        if (!in)
        {
            LOG_WARNING() << "SaveLoader::Load: failed to open layer file: " << path.string();
            return {};
        }

        LayerFile layerFile;
        layerFile.path = path;

        LayerFileEntry current;
        std::string line;
        while (std::getline(in, line))
        {
            Trim(line);
            if (line.empty())
            {
                continue;
            }

            if (line == ";")
            {
                if (!current.name.empty())
                {
                    if (current.name == "Layer_Name")
                    {
                        layerFile.rootLayerName = current.data;
                        layerFile.rootLayerType = current.type;
                        layerFile.rootLayerMeta = current.meta;
                        layerFile.rootLayerGuid = current.guid;
                        layerFile.hasRootLayerGuid = current.hasGuid;
                    }
                    layerFile.entries.push_back(current);
                }
                current = {};
                continue;
            }

            const std::string namePrefix = "NAME ";
            const std::string dataPrefix = "DATA ";
            const std::string typePrefix = "TYPE ";
            const std::string metaPrefix = "META ";
            const std::string guidPrefix = "GUID ";

            if (line.rfind(namePrefix, 0) == 0)
            {
                current.name = line.substr(namePrefix.size());
                Trim(current.name);
                continue;
            }
            if (line.rfind(dataPrefix, 0) == 0)
            {
                current.data = line.substr(dataPrefix.size());
                Trim(current.data);
                continue;
            }
            if (line.rfind(typePrefix, 0) == 0)
            {
                current.type = line.substr(typePrefix.size());
                Trim(current.type);
                continue;
            }
            if (line.rfind(metaPrefix, 0) == 0)
            {
                current.meta = line.substr(metaPrefix.size());
                Trim(current.meta);
                continue;
            }
            if (line.rfind(guidPrefix, 0) == 0)
            {
                std::string guidStr = line.substr(guidPrefix.size());
                Trim(guidStr);
                try
                {
                    current.guid = static_cast<uint64_t>(std::stoull(guidStr));
                    current.hasGuid = true;
                }
                catch (...)
                {
                    current.guid = 0u;
                    current.hasGuid = false;
                }
                continue;
            }
        }

        if (!current.name.empty())
        {
            if (current.name == "Layer_Name")
            {
                layerFile.rootLayerName = current.data;
                layerFile.rootLayerType = current.type;
                layerFile.rootLayerMeta = current.meta;
                layerFile.rootLayerGuid = current.guid;
                layerFile.hasRootLayerGuid = current.hasGuid;
            }
            layerFile.entries.push_back(current);
        }

        return layerFile;
    }

    static bool ParseOptimiserFile(const std::filesystem::path& path, OptimiserFile& outFile)
    {
        std::ifstream in(path);
        if (!in)
        {
            LOG_WARNING() << "SaveLoader::LoadOptimiser: failed to open optimiser file: " << path.string();
            return false;
        }

        bool inHeader = true;
        OptimiserParamFile currentParam;
        std::string line;
        while (std::getline(in, line))
        {
            Trim(line);
            if (line.empty())
            {
                continue;
            }

            if (line == ";")
            {
                if (inHeader)
                {
                    inHeader = false;
                }
                else if (currentParam.hasGuid || currentParam.hasM0 || currentParam.hasM1 || currentParam.hasTimestep)
                {
                    outFile.params.push_back(currentParam);
                    currentParam = {};
                }
                continue;
            }

            const std::string typePrefix = "TYPE ";
            const std::string lrPrefix = "LR ";
            const std::string beta1Prefix = "BETA1 ";
            const std::string beta2Prefix = "BETA2 ";
            const std::string epsPrefix = "EPS ";
            const std::string weightDecayPrefix = "WEIGHT_DECAY ";
            const std::string paramGuidPrefix = "PARAM_GUID ";
            const std::string m0Prefix = "M0 ";
            const std::string m1Prefix = "M1 ";
            const std::string tPrefix = "T ";

            if (inHeader)
            {
                if (line.rfind(typePrefix, 0) == 0)
                {
                    outFile.type = line.substr(typePrefix.size());
                    Trim(outFile.type);
                    continue;
                }
                if (line.rfind(lrPrefix, 0) == 0)
                {
                    const std::string value = line.substr(lrPrefix.size());
                    try
                    {
                        outFile.lr = std::stod(value);
                        outFile.hasLr = true;
                    }
                    catch (...)
                    {
                        outFile.hasLr = false;
                    }
                    continue;
                }
                if (line.rfind(beta1Prefix, 0) == 0)
                {
                    const std::string value = line.substr(beta1Prefix.size());
                    try
                    {
                        outFile.beta1 = std::stod(value);
                        outFile.hasBeta1 = true;
                    }
                    catch (...)
                    {
                        outFile.hasBeta1 = false;
                    }
                    continue;
                }
                if (line.rfind(beta2Prefix, 0) == 0)
                {
                    const std::string value = line.substr(beta2Prefix.size());
                    try
                    {
                        outFile.beta2 = std::stod(value);
                        outFile.hasBeta2 = true;
                    }
                    catch (...)
                    {
                        outFile.hasBeta2 = false;
                    }
                    continue;
                }
                if (line.rfind(epsPrefix, 0) == 0)
                {
                    const std::string value = line.substr(epsPrefix.size());
                    try
                    {
                        outFile.eps = std::stod(value);
                        outFile.hasEps = true;
                    }
                    catch (...)
                    {
                        outFile.hasEps = false;
                    }
                    continue;
                }
                if (line.rfind(weightDecayPrefix, 0) == 0)
                {
                    const std::string value = line.substr(weightDecayPrefix.size());
                    try
                    {
                        outFile.weightDecay = std::stod(value);
                        outFile.hasWeightDecay = true;
                    }
                    catch (...)
                    {
                        outFile.hasWeightDecay = false;
                    }
                    continue;
                }
                continue;
            }

            if (line.rfind(paramGuidPrefix, 0) == 0)
            {
                const std::string value = line.substr(paramGuidPrefix.size());
                try
                {
                    currentParam.paramGuid = static_cast<uint64_t>(std::stoull(value));
                    currentParam.hasGuid = true;
                }
                catch (...)
                {
                    currentParam.hasGuid = false;
                }
                continue;
            }
            if (line.rfind(m0Prefix, 0) == 0)
            {
                currentParam.m0File = line.substr(m0Prefix.size());
                Trim(currentParam.m0File);
                currentParam.hasM0 = !currentParam.m0File.empty();
                continue;
            }
            if (line.rfind(m1Prefix, 0) == 0)
            {
                currentParam.m1File = line.substr(m1Prefix.size());
                Trim(currentParam.m1File);
                currentParam.hasM1 = !currentParam.m1File.empty();
                continue;
            }
            if (line.rfind(tPrefix, 0) == 0)
            {
                const std::string value = line.substr(tPrefix.size());
                try
                {
                    currentParam.timestep = static_cast<uint64_t>(std::stoull(value));
                    currentParam.hasTimestep = true;
                }
                catch (...)
                {
                    currentParam.hasTimestep = false;
                }
                continue;
            }
        }

        if (!inHeader && (currentParam.hasGuid || currentParam.hasM0 || currentParam.hasM1 || currentParam.hasTimestep))
        {
            outFile.params.push_back(currentParam);
        }

        if (outFile.type.empty())
        {
            LOG_WARNING() << "SaveLoader::LoadOptimiser: missing TYPE in optimiser file: " << path.string();
            return false;
        }

        return true;
    }

    static bool ParseLossFile(const std::filesystem::path& path, LossFile& outFile)
    {
        std::ifstream in(path);
        if (!in)
        {
            LOG_WARNING() << "SaveLoader::LoadLoss: failed to open loss file: " << path.string();
            return false;
        }

        std::string line;
        while (std::getline(in, line))
        {
            Trim(line);
            if (line.empty())
            {
                continue;
            }

            if (line == ";")
            {
                break;
            }

            const std::string typePrefix = "TYPE ";
            if (line.rfind(typePrefix, 0) == 0)
            {
                outFile.type = line.substr(typePrefix.size());
                Trim(outFile.type);
            }
        }

        if (outFile.type.empty())
        {
            LOG_WARNING() << "SaveLoader::LoadLoss: missing TYPE in loss file: " << path.string();
            return false;
        }

        return true;
    }

    static std::string NormalizeMeta(std::string meta)
    {
        Trim(meta);
        std::transform(meta.begin(), meta.end(), meta.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return meta;
    }

    static Layer<T, Mat>* CreateDenseFromMeta(const std::string& meta)
    {
        const std::string norm = NormalizeMeta(meta);
        if (norm == "gelu")
        {
            return new Dense<T, Gelu<T, Mat>, Mat>();
        }
        if (norm == "relu")
        {
            return new Dense<T, Relu<T>, Mat>();
        }
        if (norm == "relu_opt")
        {
            return new Dense<T, ReluOpt<T>, Mat>();
        }
        if (norm == "leaky_relu")
        {
            return new Dense<T, LeakyRelu<T>, Mat>();
        }
        if (norm == "sigmoid")
        {
            return new Dense<T, Sigmoid<T>, Mat>();
        }
        if (norm == "tanh")
        {
            return new Dense<T, Tanh<T>, Mat>();
        }
        return new Dense<T, Identity<T>, Mat>();
    }

    static Layer<T, Mat>* CreateActivationLayerFromMeta(const std::string& meta)
    {
        const std::string norm = NormalizeMeta(meta);
        if (norm == "gelu")
        {
            return new ActivationLayer<T, Gelu<T, Mat>, Mat>();
        }
        if (norm == "relu")
        {
            return new ActivationLayer<T, Relu<T>, Mat>();
        }
        if (norm == "relu_opt")
        {
            return new ActivationLayer<T, ReluOpt<T>, Mat>();
        }
        if (norm == "leaky_relu")
        {
            return new ActivationLayer<T, LeakyRelu<T>, Mat>();
        }
        if (norm == "sigmoid")
        {
            return new ActivationLayer<T, Sigmoid<T>, Mat>();
        }
        if (norm == "tanh")
        {
            return new ActivationLayer<T, Tanh<T>, Mat>();
        }
        return new ActivationLayer<T, Identity<T>, Mat>();
    }

    static Layer<T, Mat>* CreateLayerFromTypeName(const std::string& typeName, const std::string& meta = "")
    {
        if (typeName == "Dense")
        {
            return CreateDenseFromMeta(meta);
        }
        if (typeName == "Conv2D")
        {
            return new Conv2D<T, Identity<T>, Mat>();
        }
        if (typeName == "ActivationLayer")
        {
            return CreateActivationLayerFromMeta(meta);
        }
        if (typeName == "AddLayer")
        {
            return new AddLayer<T, Mat>();
        }
        if (typeName == "LayerNorm")
        {
            return new LayerNorm<T, Mat>();
        }
        if (typeName == "DropoutLayer")
        {
            return new DropoutLayer<T, Mat>();
        }
        if (typeName == "Embedding")
        {
            return new Embedding<T, Mat>();
        }
        if (typeName == "CausalSelfAttentionLayer")
        {
            return new CausalSelfAttentionLayer<T, Identity<T>, Mat>();
        }
        if (typeName == "TransformerBlock")
        {
            return new TransformerBlockLayer<T, Mat>();
        }
        if (typeName == "FlattenCopy")
        {
            return new FlattenCopyLayer<T, Mat>();
        }
        return nullptr;
    }

    static Layer<T, Mat>* CreateLayerFromRootType(const LayerFile& file)
    {
        std::string rootType = file.rootLayerType;
        const std::string rootPrefix = "ROOTLAYER_";
        if (rootType.rfind(rootPrefix, 0) == 0)
        {
            rootType = rootType.substr(rootPrefix.size());
        }
        if (rootType.empty())
        {
            rootType = file.rootLayerName;
        }

        return CreateLayerFromTypeName(rootType, file.rootLayerMeta);
    }
    
    static bool IsLayerType(const std::string& type)
    {
        return type.rfind("LAYER_", 0) == 0 || type.rfind("ROOTLAYER_", 0) == 0;
    }

    template<typename ValueT>
    static bool ReadBinaryValue(const std::filesystem::path& path, ValueT& outValue)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            LOG_WARNING() << "SaveLoader::Load: failed to open binary file: " << path.string();
            return false;
        }
        in.read(reinterpret_cast<char*>(&outValue), sizeof(ValueT));
        if (!in)
        {
            LOG_WARNING() << "SaveLoader::Load: failed to read binary value from file: " << path.string();
        }
        return static_cast<bool>(in);
    }

    static bool ReadBinaryFile(const std::filesystem::path& path, std::vector<char>& outData)
    {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in)
        {
            LOG_WARNING() << "SaveLoader::Load: failed to open binary file: " << path.string();
            return false;
        }
        const std::streamsize size = in.tellg();
        if (size <= 0)
        {
            outData.clear();
            return size == 0;
        }
        outData.resize(static_cast<size_t>(size));
        in.seekg(0, std::ios::beg);
        in.read(outData.data(), size);
        if (!in)
        {
            LOG_WARNING() << "SaveLoader::Load: failed to read binary file: " << path.string();
        }
        return static_cast<bool>(in);
    }

    struct MatrixFile
    {
        std::string internalName;
        Dims3D dims;
        std::string dataFile;
        uint64_t guid = 0u;
        bool hasGuid = false;
    };

    static bool ParseMatrixFile(const std::filesystem::path& path, MatrixFile& outFile)
    {
        std::ifstream in(path);
        if (!in)
        {
            LOG_WARNING() << "SaveLoader::Load: failed to open matrix file: " << path.string();
            return false;
        }

        std::string line;
        while (std::getline(in, line))
        {
            Trim(line);
            if (line.empty())
            {
                continue;
            }

            if (line == ";")
            {
                break;
            }

            const std::string namePrefix = "INTERNAL_NAME ";
            const std::string dimxPrefix = "DIMX ";
            const std::string dimyPrefix = "DIMY ";
            const std::string dimzPrefix = "DIMZ ";
            const std::string dataPrefix = "DATA ";
            const std::string guidPrefix = "GUID ";

            if (line.rfind(namePrefix, 0) == 0)
            {
                outFile.internalName = line.substr(namePrefix.size());
                Trim(outFile.internalName);
                continue;
            }
            if (line.rfind(dimxPrefix, 0) == 0)
            {
                outFile.dims.x = static_cast<uint32_t>(std::stoul(line.substr(dimxPrefix.size())));
                continue;
            }
            if (line.rfind(dimyPrefix, 0) == 0)
            {
                outFile.dims.y = static_cast<uint32_t>(std::stoul(line.substr(dimyPrefix.size())));
                continue;
            }
            if (line.rfind(dimzPrefix, 0) == 0)
            {
                outFile.dims.z = static_cast<uint32_t>(std::stoul(line.substr(dimzPrefix.size())));
                continue;
            }
            if (line.rfind(dataPrefix, 0) == 0)
            {
                outFile.dataFile = line.substr(dataPrefix.size());
                Trim(outFile.dataFile);
                continue;
            }
            if (line.rfind(guidPrefix, 0) == 0)
            {
                std::string guidStr = line.substr(guidPrefix.size());
                Trim(guidStr);
                try
                {
                    outFile.guid = static_cast<uint64_t>(std::stoull(guidStr));
                    outFile.hasGuid = true;
                }
                catch (...)
                {
                    outFile.guid = 0u;
                    outFile.hasGuid = false;
                }
                continue;
            }
        }

        if (outFile.dataFile.empty())
        {
            LOG_WARNING() << "SaveLoader::Load: matrix file missing DATA entry: " << path.string();
            return false;
        }
        return true;
    }

    static MatrixRef LoadMatrixFromFile(const std::filesystem::path& matrixFilePath)
    {
        MatrixFile mf;
        if (!ParseMatrixFile(matrixFilePath, mf))
        {
            return MatrixRef();
        }

        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();
        if (mf.hasGuid)
        {
            if (auto* existing = inst.GetMatrixByGuid(mf.guid))
            {
                return inst.Acquire(existing);
            }
        }

        std::vector<char> raw;
        const std::filesystem::path dataPath = matrixFilePath.parent_path() / mf.dataFile;
        if (!ReadBinaryFile(dataPath, raw))
        {
            return MatrixRef();
        }

        auto mat = inst.AllocateMatrix(mf.dims, mf.internalName);
        if (!mat.get())
        {
            LOG_WARNING() << "SaveLoader::Load: failed to allocate matrix: " << matrixFilePath.string();
            return MatrixRef();
        }
        if (mf.hasGuid)
        {
            mat->m_guid = mf.guid;
        }

        const size_t expectedBytes = static_cast<size_t>(mf.dims.x * mf.dims.y * mf.dims.z * sizeof(T));
        const size_t copyBytes = std::min(expectedBytes, raw.size());
        if (copyBytes > 0)
        {
            std::memcpy(mat->DataWrite(), raw.data(), copyBytes);
        }
        if (raw.size() != expectedBytes)
        {
            LOG_WARNING() << "SaveLoader::Load: matrix data size mismatch for " << matrixFilePath.string()
                          << " expected " << expectedBytes << " bytes, got " << raw.size();
        }
        return mat;
    }

    static void PopulateBlobFromEntries(Datablob<T, Mat>* blob, const LayerFile& file)
    {
        if (!blob)
        {
            return;
        }

        const std::filesystem::path baseDir = file.path.parent_path();
        for (const auto& entry : file.entries)
        {
            if (entry.type.empty() || IsLayerType(entry.type))
            {
                continue;
            }

            if (entry.type == "INT")
            {
                int value = 0;
                if (ReadBinaryValue(baseDir / entry.data, value))
                {
                    blob->Set(entry.name, value);
                }
                else
                {
                    LOG_WARNING() << "SaveLoader::Load: failed to read INT for key: " << entry.name;
                }
                continue;
            }
            if (entry.type == "UINT")
            {
                uint32_t value = 0;
                if (ReadBinaryValue(baseDir / entry.data, value))
                {
                    blob->Set(entry.name, value);
                }
                else
                {
                    LOG_WARNING() << "SaveLoader::Load: failed to read UINT for key: " << entry.name;
                }
                continue;
            }
            if (entry.type == "FLOAT")
            {
                float value = 0.0f;
                if (ReadBinaryValue(baseDir / entry.data, value))
                {
                    blob->Set(entry.name, value);
                }
                else
                {
                    LOG_WARNING() << "SaveLoader::Load: failed to read FLOAT for key: " << entry.name;
                }
                continue;
            }
            if (entry.type == "MATRIX")
            {
                MatrixRef mat = LoadMatrixFromFile(baseDir / entry.data);
                if (mat)
                {
                    LOG_INFO() << "Loaded Matrix: " << mat.get()->GetID() << ". " << mat.get()->GetName();
                    blob->Set(entry.name, mat);
                }
                else
                {
                    LOG_WARNING() << "SaveLoader::Load: failed to read MATRIX for key: " << entry.name;
                }
                continue;
            }
            LOG_WARNING() << "SaveLoader::Load: unsupported entry type '" << entry.type
                          << "' for key: " << entry.name;
        }
    }

    static void PopulateLayerEntries(Datablob<T, Mat>* parentBlob, const LayerFile& file)
    {
        if (!parentBlob)
        {
            return;
        }

        const std::filesystem::path baseDir = file.path.parent_path();
        for (const auto& entry : file.entries)
        {
            if (entry.type.rfind("LAYER_", 0) != 0)
            {
                continue;
            }

            const std::filesystem::path layerPath = baseDir / entry.data;
            LayerFile childFile = ParseLayerFile(layerPath);
            if (childFile.path.empty())
            {
                LOG_WARNING() << "SaveLoader::Load: failed to parse child layer file: " << layerPath.string();
                continue;
            }

            Layer<T, Mat>* childLayer = CreateLayerFromRootType(childFile);
            if (!childLayer)
            {
                LOG_WARNING() << "SaveLoader::Load: unsupported child layer type: " << childFile.rootLayerType;
                continue;
            }

            if (!childFile.rootLayerName.empty())
            {
                childLayer->SetName(childFile.rootLayerName);
            }
            if (childFile.hasRootLayerGuid)
            {
                childLayer->m_guid = childFile.rootLayerGuid;
            }

            Datablob<T, Mat>* childBlob = new Datablob<T, Mat>();
            PopulateBlobFromEntries(childBlob, childFile);

            LOG_INFO() << "Setting Layer" << entry.name << " From " << childLayer;
            LOG_INFO() << "Setting Blob" << entry.name << " From " << childBlob;
            parentBlob->Set(entry.name, childLayer);
            parentBlob->Set(entry.name, childBlob);

            PopulateLayerEntries(childBlob, childFile);
        }
    }

    std::string WriteMatrixFile(MatrixBase<T>* _mat)
    {
        const std::string guid = GenerateGuid();
        const std::string filename = guid + ".blob_matrix";
        std::ofstream out(filename, std::ios::out);
        if (!out)
        {
            return {};
        }

        std::stringstream ss;
        std::vector<char> _rawData;
        _mat->GetRawData(_rawData);
        std::string datafile = WriteBinaryDataFile(_rawData.data(), _rawData.size());
        ss << "INTERNAL_NAME " << _mat->GetName() << "\n";
        ss << "GUID " << _mat->m_guid << "\n";
        ss << "DIMX " << _mat->GetDimsX() << "\n";
        ss << "DIMY " << _mat->GetDimsY() << "\n";
        ss << "DIMZ " << _mat->GetDimsZ() << "\n";
        ss << "DATA " << datafile << "\n";
        ss << ";\n";
        out.write(ss.str().c_str(), static_cast<std::streamsize>(ss.str().length()));
        return filename; 
    }
    std::string WriteBinaryDataFile(const void* _data, size_t _size)
    {
        const std::string guid = GenerateGuid();
        const std::string filename = guid + ".blob_data";
        std::ofstream out(filename, std::ios::binary);
        if (!out)
        {
            return {};
        }

        if (_data && _size > 0u)
        {
            out.write(reinterpret_cast<const char*>(_data),
                      static_cast<std::streamsize>(_size));
        }
        return filename;
    }


    std::string WriteBlobFile(const std::string& _data)
    {
        const std::string guid = GenerateGuid();
        const std::string filename = guid + ".blob_layer";
        std::ofstream out(filename, std::ios::out);
        if (!out)
        {
            return {};
        }

        if (!_data.empty())
        {
            out.write(_data.data(), static_cast<std::streamsize>(_data.size()));
        }
        return filename;
    }

    std::string WriteOptimiserFile(const std::string& _data)
    {
        const std::string guid = GenerateGuid();
        const std::string filename = guid + ".blob_optimiser";
        std::ofstream out(filename, std::ios::out);
        if (!out)
        {
            return {};
        }

        if (!_data.empty())
        {
            out.write(_data.data(), static_cast<std::streamsize>(_data.size()));
        }
        return filename;
    }

    std::string WriteLossFile(const std::string& _data)
    {
        const std::string guid = GenerateGuid();
        const std::string filename = guid + ".blob_loss";
        std::ofstream out(filename, std::ios::out);
        if (!out)
        {
            return {};
        }

        if (!_data.empty())
        {
            out.write(_data.data(), static_cast<std::streamsize>(_data.size()));
        }
        return filename;
    }
};
