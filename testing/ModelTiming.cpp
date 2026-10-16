#include "LayerTestSupport.h"
#include <ModelLibrary/Model.h>

using namespace LayerTesting;
int main()
{
    try {
        LOG_INFO() << "Starting model timing test";
#if defined(LAYER_TEST_Metal)
        if (!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if (!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        using M = TestMatrix;
        MatrixShape dims = MatrixShape::Matrix(32,16);
        auto& manager = MatrixManager<float,M>::Instance();
        auto input = manager.AllocateMatrix(dims,"TimingInput");
        auto target = manager.AllocateMatrix(dims,"TimingTarget");
        auto values = Pattern(input->GetElementCount());
        Write(input.get(),values);
        Write(target.get(),Values(values.size(),0.0f));
        Model<float,M> model;
        std::string firstName="First", secondName="Second";
        auto* first=new ActivationLayer<float,Identity<float>,M>();
        auto* second=new ActivationLayer<float,Identity<float>,M>();
        first->SetName(firstName);second->SetName(secondName);
        auto* firstBlob=InitActivationBlob<float,M>(dims,firstName);
        auto* secondBlob=InitActivationBlob<float,M>(dims,secondName);
        model.AddLayer(first,firstBlob);model.AddLayer(second,secondBlob);
        model.AddDependency(second,0,first,0);
        model.SetFinalOutputLayer(second);model.SetExpectedInputDims(dims);model.Init();
        for(bool sync : {false,true,false}) {
            model.SetTimingSynchronization(sync);
            model.Run(input.get(),target.get());
#if defined(LAYER_TEST_Metal)
            auto* gpu=MetalManager::Instance();
            bool pending=gpu->m_workToCommit || gpu->m_lastSubmitted!=nullptr;
            Require(pending!=sync,"Unexpected completion state after timed model run");
#endif
            Expect(Read(model.GetOutput()),values,"Timed model output");
            auto gradient=Read(first->GetOutputError(firstBlob).get());
            auto expected=values;for(auto& value:expected) value=-value;
            Expect(gradient,expected,"Timed model gradient");
            model.PrintTimings();
        }
        std::cout<<"PASS: model timing synchronization\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
