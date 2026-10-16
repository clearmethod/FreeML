#include "LayerTestSupport.h"
#include <ModelLibrary/Model.h>
#include <OptimiserLibrary/AdamWOptimiser.h>
#include <LossLibrary/CategoricalCrossEntropyWithLogits.h>

using namespace LayerTesting;
using M=TestMatrix;
using Manager=MatrixManager<float,M>;

struct TrainingModel {
    std::unique_ptr<Model<float,M>> model=std::make_unique<Model<float,M>>();
    Manager::MatrixRef weights;
    AdamWOptimiser<float,M>* optimizer=nullptr;
    explicit TrainingModel(unsigned batch) {
        auto* left=new Dense<float,Identity<float>,M>;
        auto* right=new Dense<float,Identity<float>,M>;
        auto* a=InitDenseBlob<float,M>(3,2,2,true,false,false);
        auto* b=InitDenseBlob<float,M>(3,2,2,true,false,false);
        weights=a->AcquireMatrix("Dense_Weights");b->Set("Dense_Weights",weights);
        model->AddLayer(left,a);model->AddLayer(right,b);
        auto* sum=new AddLayer<float,M>;
        model->AddLayer(sum,InitAddBlob<float,M>());
        model->AddDependency(sum,0,left,0);model->AddDependency(sum,1,right,0);
        model->SetFinalOutputLayer(sum);model->SetExpectedInputDims(MatrixShape::Sequence(batch,2,2));model->Init();
        Write(weights.get(),{0.2f,-0.1f,0.4f,0.3f,0.1f,-0.2f});
        optimizer=new AdamWOptimiser<float,M>(0.01f,0.9f,0.99f,1e-8f,0.01f);
        model->SetOptimiser(optimizer);model->SetLoss(new CategoricalCrossEntropyWithLogits<float,M>);
        model->SetGradientAccumulationSteps(12u/batch);model->SetGradientClipNorm(1.0f);
        model->SetSelectiveWeightDecay(true);
    }
};
int main()
{
    try {
        LOG_INFO()<<"Starting physical batch/update equivalence tests";
        Tools::Logger::SetMinLevel(Tools::Logger::Level::Warning);
#if defined(LAYER_TEST_Metal)
        if(!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if(!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        auto& manager=Manager::Instance();
        auto input=manager.AllocateMatrix(MatrixShape::Sequence(12,2,2),"UpdateInputs");
        auto target=manager.AllocateMatrix(MatrixShape::Sequence(12,1,2),"UpdateTargets");
        Write(input.get(),Pattern(48,0.8f));Values labels(24);
        for(unsigned i=0;i<labels.size();++i) labels[i]=float((i*7+i/3)%3);
        Write(target.get(),labels);
        TrainingModel single(1),batched(3);
        bool rejected=false;
        try { single.model->Run(input.get(),nullptr); }
        catch(const std::invalid_argument&) { rejected=true; }
        Require(rejected,"Model must reject shape changes before rebinding initialized graph buffers");
        for(unsigned update=0;update<3;++update) {
            for(auto* fixture:{&single,&batched}) {
                const unsigned batch=fixture==&single?1u:3u;
                fixture->model->BeginLossBatch();
                for(unsigned sample=0;sample<12;sample+=batch) {
                    auto x=manager.CreateView(input,MatrixShape::Sequence(batch,2,2),sample*4);
                    auto y=manager.CreateView(target,MatrixShape::Sequence(batch,1,2),sample*2);
                    fixture->model->Run(x.get(),y.get());
                }
                fixture->model->EndLossBatch();
            }
            Near(batched.model->GetLastLoss(),single.model->GetLastLoss(),"Equivalent mean loss");
            Expect(Read(batched.weights.get()),Read(single.weights.get()),"Equivalent clipped AdamW update");
        }
        std::vector<AdamWOptimiser<float,M>::ParamState> a,b;
        single.optimizer->ExportState(a);batched.optimizer->ExportState(b);
        Require(a.size()==1u && b.size()==1u && a[0].timestep==3u && b[0].timestep==3u,"Tied parameter update count");
        Expect(Read(b[0].moment0.get()),Read(a[0].moment0.get()),"Equivalent first moments");
        Expect(Read(b[0].moment1.get()),Read(a[0].moment1.get()),"Equivalent second moments");
        std::cout<<"PASS: physical batching, accumulated means, tied weights, clipping and AdamW\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
}
