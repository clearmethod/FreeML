#include "LayerTestSupport.h"
#include <ModelLibrary/Model.h>
#include <OptimiserLibrary/AdamWOptimiser.h>
#include <LossLibrary/CategoricalCrossEntropyWithLogits.h>
#include <random>
#include <unordered_set>

using namespace LayerTesting;
int main(int argc, char** argv)
{
    try {
        LOG_INFO() << "Starting NanoLLM convergence regression";
#if defined(LAYER_TEST_Metal)
        if(!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if(!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        using M=TestMatrix;
        constexpr uint32_t width=24,tokens=16,vocab=8;
        unsigned physicalBatch=1u;
        for(int i=1;i<argc;++i) if(std::string(argv[i])=="--physical-batch") physicalBatch=3u;
        auto& manager=MatrixManager<float,M>::Instance();
        const auto tokenShape=MatrixShape::Sequence(physicalBatch,1u,tokens);
        auto input=manager.AllocateMatrix(tokenShape,"Tokens");
        auto target=manager.AllocateMatrix(tokenShape,"Targets");
        auto positions=manager.AllocateMatrix(tokenShape,"Positions");
        Values ids(tokens*physicalBatch),next(tokens*physicalBatch),pos(tokens*physicalBatch);
        for(uint32_t i=0;i<ids.size();++i) {ids[i]=float(i%vocab);next[i]=float((i+1)%vocab);pos[i]=float(i%tokens);}
        Write(input.get(),ids);Write(target.get(),next);Write(positions.get(),pos);
        Model<float,M> model;
        std::vector<std::pair<Layer<float,M>*,Datablob<float,M>*>> layers;
        auto add=[&](auto* layer,auto* blob,const char* name) {
            layer->SetName(name);model.AddLayer(layer,blob);layers.emplace_back(layer,blob);return layer;
        };
        auto* tokenBlob=InitEmbeddingBlob<float,M>(vocab,width,tokens,1,true,false);
        auto* token=add(new Embedding<float,M>,tokenBlob,"Token");
        auto* positionBlob=InitEmbeddingBlob<float,M>(tokens,width,tokens,1,true,false);
        auto* position=add(new Embedding<float,M>,positionBlob,"Position");
        auto* sum=add(new AddLayer<float,M>,InitAddBlob<float,M>(),"Sum");
        model.AddDependency(sum,0,token,0);model.AddDependency(sum,1,position,0);
        auto* dropout=add(new DropoutLayer<float,M>,InitDropoutBlob<float,M>(0,true),"Dropout");
        model.AddDependency(dropout,0,sum,0);
        auto* block=add(new TransformerBlockLayer<float,M>,
                       InitTransformerBlockBlob<float,M>(width,6,0,tokens,1,true,false),"Block");
        model.AddDependency(block,0,dropout,0);
        auto* norm=add(new LayerNorm<float,M>,InitLayerNormBlob<float,M>(true,false,false),"Norm");
        model.AddDependency(norm,0,block,0);
        auto* headBlob=InitDenseBlob<float,M>(vocab,width,tokens,true,false,false,true);
        auto oldWeight=headBlob->AcquireMatrix("Dense_Weights");
        headBlob->Set("Dense_Weights",tokenBlob->AcquireMatrix("Embedding_Weights"));
        manager.RemoveMatrix(oldWeight.get());oldWeight.Reset();
        auto* head=add(new Dense<float,Identity<float>,M,true>,headBlob,"Head");
        model.AddDependency(head,0,norm,0);model.SetFinalOutputLayer(head);
        model.SetExpectedInputDims(tokenShape);model.Init();
        position->SetInput(positionBlob,positions);
        // Identical deterministic initialization on every backend, including tied weights.
        std::mt19937 rng(1234);std::normal_distribution<float> normal(0,0.02f);
        std::unordered_set<M*> initialized;
        for(auto [layer,blob]:layers) for(auto& weight:*layer->GetWeights(blob)) {
            if(!initialized.insert(weight.get()).second) continue;
            Values values(weight->GetElementCount());
            for(auto& v:values) v=weight->GetName()=="GammaScale" ? 1.0f : normal(rng);
            Write(weight.get(),values);
        }
        Require(headBlob->GetMatrix("Dense_Weights")==tokenBlob->GetMatrix("Embedding_Weights"),"Output weights must be tied");
        auto* optimizer=new AdamWOptimiser<float,M>(0.01f,0.9f,0.999f,1e-8f,0.01f);
        model.SetOptimiser(optimizer);model.SetLoss(new CategoricalCrossEntropyWithLogits<float,M>);
        bool accumulate = false, selectiveDecay = false, clip = false, batchLoss = false;
        for (int i = 1; i < argc; ++i) {
            accumulate |= std::string(argv[i]) == "--accumulate";
            selectiveDecay |= std::string(argv[i]) == "--selective-decay";
            clip |= std::string(argv[i]) == "--clip";
            batchLoss |= std::string(argv[i]) == "--batch-loss";
        }
        const unsigned accumulationSteps = accumulate ? 4u : 1u;
        model.SetGradientAccumulationSteps(accumulationSteps);
        model.SetSelectiveWeightDecay(selectiveDecay);
        model.SetGradientClipNorm(clip ? 1.0f : 0.0f);
        double first=0,last=0;
        for(unsigned step=0;step<100;++step) {
            if (batchLoss) model.BeginLossBatch();
            for (unsigned pass = 0; pass < accumulationSteps; ++pass)
                model.Run(input.get(),target.get(),!batchLoss);
            last=batchLoss ? model.EndLossBatch() : model.GetLastLoss();if(step==0) first=last;
            Require(std::isfinite(last),"Training loss became non-finite");
            if(step%20==0 || step==99) std::cout<<"step "<<step<<" loss "<<last<<'\n';
        }
        auto logits=Read(model.GetOutput());unsigned correct=0;
        for(uint32_t row=0;row<tokens*physicalBatch;++row) {
            auto begin=logits.begin()+row*vocab;
            correct+=uint32_t(std::max_element(begin,begin+vocab)-begin)==uint32_t(next[row]);
        }
        Require(last<first*0.25,"Tiny NanoLLM must reduce its loss by at least 75%");
        Require(correct>=tokens*physicalBatch*0.9,"Tiny NanoLLM must learn the fixed next-token sequence");
        std::vector<typename AdamWOptimiser<float,M>::ParamState> states;optimizer->ExportState(states);
        Require(states.size() == initialized.size(), "Every unique parameter must be updated");
        Require(model.GetPendingGradientPasses() == 0u, "Training must end with a complete update");
        for(auto& state:states) Require(state.timestep==100,"Tied parameters must update once per training step");
        std::cout<<"PASS: NanoLLM loss "<<first<<" -> "<<last<<", accuracy "<<float(correct)/tokens<<'\n';
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
