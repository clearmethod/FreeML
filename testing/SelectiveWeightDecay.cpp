#include "LayerTestSupport.h"
#include <ModelLibrary/Model.h>
#include <OptimiserLibrary/AdamWOptimiser.h>

using namespace LayerTesting;
using M = TestMatrix;

class DecayOnlyOptimiser : public Optimiser<float,M>
{
public:
    std::unordered_map<M*,float> scales;
    void Step(M* parameter, M* gradient) override { Step(parameter,gradient,1.0f); }
    void Step(M* parameter, M*, float scale) override
    {
        Require(scales.emplace(parameter,scale).second,"Tied parameter stepped twice");
        auto values = Read(parameter);
        for (auto& value : values) value *= 1.0f-0.001f*scale;
        Write(parameter,values);
    }
};

void CheckModel(bool selective, unsigned passes, bool partial, bool conflicting = false)
{
    auto& manager = MatrixManager<float,M>::Instance();
    auto input = manager.AllocateMatrix(MatrixShape::Matrix(1,1),"DecayInput");
    auto target = manager.AllocateMatrix(MatrixShape::Matrix(1,1),"DecayTarget");
    Write(input.get(),{1}); Write(target.get(),{0});
    Model<float,M> model;
    auto* first = new Dense<float,Identity<float>,M>;
    auto* second = new Dense<float,Identity<float>,M>;
    auto* fb = InitDenseBlob<float,M>(1,1,1,true,false,true);
    auto* sb = InitDenseBlob<float,M>(1,1,1,true,false,true);
    auto weight = fb->AcquireMatrix(conflicting ? "Dense_Bias" : "Dense_Weights");
    auto oldWeight = sb->AcquireMatrix("Dense_Weights");
    sb->Set("Dense_Weights",weight); manager.RemoveMatrix(oldWeight);
    auto b1 = fb->AcquireMatrix("Dense_Bias"), b2 = sb->AcquireMatrix("Dense_Bias");
    Write(weight.get(),{1}); Write(b1.get(),{2}); Write(b2.get(),{2});
    model.AddLayer(first,fb); model.AddLayer(second,sb); model.AddDependency(second,0,first,0);
    model.SetExpectedInputDims(MatrixShape::Matrix(1,1)); model.SetFinalOutputLayer(second); model.Init();
    auto* optimizer = new DecayOnlyOptimiser;
    model.SetOptimiser(optimizer);
    Require(!model.GetSelectiveWeightDecay(),"Selective decay must be opt-in");
    model.SetSelectiveWeightDecay(selective);
    model.SetGradientAccumulationSteps(partial ? passes+1 : passes);
    for (unsigned i=0;i<passes;++i) {
        try { model.Run(input.get(),target.get()); }
        catch (const std::invalid_argument&) {
            if (!conflicting) throw;
            Require(optimizer->scales.empty(),"Conflicting decay policy reached the optimizer");
            return;
        }
        if (i+1<passes || partial) Require(optimizer->scales.empty(),"Premature decay update");
    }
    Require(!conflicting,"Conflicting tied decay policies accepted");
    // A policy change must flush pending gradients using the old settings.
    if (partial) model.SetSelectiveWeightDecay(!selective);
    Require(optimizer->scales.size()==3,"Expected one weight and two biases");
    Near(optimizer->scales.at(weight.get()),1,"Weight decay");
    Near(optimizer->scales.at(b1.get()),selective ? 0 : 1,"First bias decay");
    Near(optimizer->scales.at(b2.get()),selective ? 0 : 1,"Second bias decay");
    Expect(Read(weight.get()),{0.999f},"Shared weight decayed once");
    Expect(Read(b1.get()),{selective ? 2.0f : 1.998f},"Bias preservation");
}

int main()
{
    try {
        LOG_INFO() << "Starting selective weight decay tests";
#if defined(LAYER_TEST_Metal)
        if (!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if (!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        for (bool selective : {false,true}) {
            CheckModel(selective,1,false);
            CheckModel(selective,3,false);
            CheckModel(selective,2,true);
        }
        CheckModel(true,1,false,true);
        // Nested attention and MLP Dense parameters must inherit their leaf
        // policies, even though Model sees a single TransformerBlock layer.
        Fixture<M> transformer("TransformerBlock");
        const Values expectedScales{0,0,1,0,1,0,0,0,1,0,1,0};
        auto* weights = transformer.layer->GetWeights(transformer.blob.get());
        Require(weights->size()==expectedScales.size(),"Unexpected transformer parameter count");
        for (size_t i=0;i<weights->size();++i)
            Near(transformer.layer->GetWeightDecayScale(transformer.blob.get(),(*weights)[i].get()),
                 expectedScales[i],"Nested parameter decay");

        auto& manager = MatrixManager<float,M>::Instance();
        auto gradient = manager.AllocateMatrix(MatrixShape::Matrix(1,1),"ZeroDecayGradient");
        Write(gradient.get(),{0});
        AdamWOptimiser<float,M> adam(0.01f,0.9f,0.99f,1e-8f,0.1f);
        for (float scale : {0.0f,0.5f,1.0f}) {
            auto parameter = manager.AllocateMatrix(MatrixShape::Matrix(1,1),"DecayParameter");
            Write(parameter.get(),{2});
            adam.Step(parameter.get(),gradient.get(),scale);
            Expect(Read(parameter.get()),{2.0f*(1.0f-0.001f*scale)},"AdamW decay multiplier");
        }
        std::cout << "PASS: selective decay, nested layers, tied parameters and accumulation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
