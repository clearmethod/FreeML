#include "LayerTestSupport.h"
#include <ModelLibrary/Model.h>
#include <LossLibrary/CategoricalCrossEntropyWithLogits.h>

using namespace LayerTesting;
using M = TestMatrix;

class CountUpdates : public Optimiser<float,M>
{
public:
    unsigned count=0;
    void Step(M*,M*) override { ++count; }
};

int main()
{
    try {
        LOG_INFO() << "Starting deferred batch loss tests";
#if defined(LAYER_TEST_Metal)
        if(!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if(!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        auto& manager=MatrixManager<float,M>::Instance();
        const Values inputs{1,2,3,4, -1,2,0,3, 2,-2,1,0, 4,1,-3,2};
        const Values targets{0,1, 2,0, 1,2, 2,1};
        const Values weights{0.5f,-0.3f,0.2f, 1.2f,0.7f,-0.5f};
        auto input=manager.AllocateMatrix(MatrixShape::Image(1u,4,2,2),"BatchLossInput");
        auto target=manager.AllocateMatrix(MatrixShape::Image(1u,4,1,2),"BatchLossTarget");
        Write(input.get(),inputs);Write(target.get(),targets);
        Model<float,M> model;
        auto* dense=new Dense<float,Identity<float>,M>;
        auto* blob=InitDenseBlob<float,M>(3,2,2,true,false,false);
        model.AddLayer(dense,blob);model.SetFinalOutputLayer(dense);
        model.SetExpectedInputDims(MatrixShape::Matrix(2,2));model.Init();
        Write(blob->GetMatrix("Dense_Weights"),weights);
        model.SetLoss(new CategoricalCrossEntropyWithLogits<float,M>);
        auto* optimizer=new CountUpdates;model.SetOptimiser(optimizer);
        model.SetGradientAccumulationSteps(4);model.SetRollingLossWindow(2);
        Values losses;
        CategoricalCrossEntropyWithLogits<float,MatrixCPU<float>> cpu;
        for(unsigned pass=0;pass<4;++pass) {
            Values logits(6),labels(targets.begin()+2*pass,targets.begin()+2*pass+2);
            for(unsigned row=0;row<2;++row) for(unsigned col=0;col<3;++col)
                logits[row*3+col]=inputs[pass*4+row*2]*weights[col]+inputs[pass*4+row*2+1]*weights[3+col];
            MatrixCPU<float> prediction(logits.data(),MatrixShape::Matrix(3,2)),truth(labels.data(),MatrixShape::Matrix(1,2));
            losses.push_back(cpu.Loss(&truth,&prediction));
        }
#if defined(LAYER_TEST_Metal)
        auto* gpu=MetalManager::Instance();gpu->WaitForGPU();
        Obj cb=gpu->GetCommandBuffer();
#endif
        model.BeginLossBatch();
        bool rejected=false;
        try { model.BeginLossBatch(); } catch(const std::logic_error&) { rejected=true; }
        Require(rejected,"Nested loss batches accepted");
        M iv,tv;
        for(unsigned pass=0;pass<4;++pass) {
            input->GetPlaneView(&iv,pass);target->GetPlaneView(&tv,pass);
            model.Run(&iv,&tv,true);
            Near(model.GetLastLoss(),0,"Deferred loss must not publish a sequence loss");
        }
        Require(optimizer->count==1,"Optimizer update must precede loss readback");
#if defined(LAYER_TEST_Metal)
        Require(gpu->GetCommandBuffer()==cb && !gpu->m_lastSubmitted,"Sequence loss introduced a completion boundary");
        Require(!gpu->GetMappedPointer(input->m_dataHandles.m_bufferHandle)
            && !gpu->GetMappedPointer(target->m_dataHandles.m_bufferHandle)
            && !gpu->GetMappedPointer(model.GetOutput()->m_dataHandles.m_bufferHandle),"Batch loss mapped GPU inputs or logits");
#endif
        const float mean=(losses[0]+losses[1]+losses[2]+losses[3])/4;
        Near(float(model.EndLossBatch()),mean,"Batch mean");
        Near(model.GetLastLoss(),mean,"Published batch loss");
        Near(float(model.GetRollingLoss()),mean,"Rolling batch mean");
        model.BeginLossBatch();
        for(unsigned pass=1;pass<3;++pass) {
            input->GetPlaneView(&iv,pass);target->GetPlaneView(&tv,pass);model.Run(&iv,&tv);
        }
        const float partialMean=(losses[1]+losses[2])/2;
        Near(float(model.EndLossBatch()),partialMean,"Actual sample count in partial loss batch");
        Require(optimizer->count==1,"Loss batch completion must not flush parameter gradients");
        model.FlushAccumulatedGradients();Require(optimizer->count==2,"Partial gradient update failed");
        Near(float(model.GetRollingLoss()),(mean+partialMean)/2,"Rolling completed batches");
        // Legacy immediate reporting remains available outside a batch.
        input->GetPlaneView(&iv,0);target->GetPlaneView(&tv,0);model.Run(&iv,&tv,true);
        Near(model.GetLastLoss(),losses[0],"Immediate loss reporting");
        Near(float(model.GetRollingLoss()),(partialMean+losses[0])/2,"Rolling window eviction");
        std::cout << "PASS: deferred means, update ordering, rolling history and immediate reporting\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
}
