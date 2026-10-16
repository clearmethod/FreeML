#include "LayerTestSupport.h"
#include <ModelLibrary/Model.h>

using namespace LayerTesting;
using M = TestMatrix;

class CaptureOptimiser : public Optimiser<float,M>
{
public:
    std::unordered_map<M*,Values> gradients;
    void Step(M* parameter, M* gradient) override
    {
        gradients[parameter] = Read(gradient);
    }
};

void CheckClipping(bool tied, unsigned passes, bool partial, float inputValue, float limit)
{
    auto& manager = MatrixManager<float,M>::Instance();
    auto input = manager.AllocateMatrix(MatrixShape::Matrix(1,1),"ClipInput");
    auto target = manager.AllocateMatrix(MatrixShape::Matrix(1,1),"ClipTarget");
    Write(input.get(),{inputValue}); Write(target.get(),{0.0f});
    Model<float,M> model;
    auto* first = new Dense<float,Identity<float>,M>;
    auto* second = new Dense<float,Identity<float>,M>;
    auto* fb = InitDenseBlob<float,M>(1,1,1,true,false,false);
    auto* sb = InitDenseBlob<float,M>(1,1,1,true,false,false);
    auto fw = fb->AcquireMatrix("Dense_Weights");
    if (tied) {
        auto old = sb->AcquireMatrix("Dense_Weights");
        sb->Set("Dense_Weights",fw); manager.RemoveMatrix(old);
    }
    auto sw = sb->AcquireMatrix("Dense_Weights");
    Write(fw.get(),{tied ? 2.0f : 1.0f}); Write(sw.get(),{2.0f});
    model.AddLayer(first,fb); model.AddLayer(second,sb);
    model.AddDependency(second,0,first,0);
    model.SetExpectedInputDims(MatrixShape::Matrix(1,1)); model.SetFinalOutputLayer(second); model.Init();
    auto* capture = new CaptureOptimiser;
    model.SetOptimiser(capture);
    Require(model.GetGradientClipNorm()==0.0f,"Clipping must default to disabled");
    model.SetGradientClipNorm(limit);
    model.SetGradientAccumulationSteps(partial ? passes+1 : passes);
    for (unsigned i=0;i<passes;++i) {
        model.Run(input.get(),target.get());
        if (i+1 < passes || partial)
            Require(capture->gradients.empty(),"Clipping must not trigger an early update");
    }
    if (partial) model.FlushAccumulatedGradients();
    // Output = w1*w2*x, with negative DirectLoss gradients.
    const double g1 = (tied ? -16.0 : -4.0)*inputValue*inputValue;
    const double g2 = -2.0*inputValue*inputValue;
    const double norm = tied ? std::abs(g1) : std::sqrt(g1*g1+g2*g2);
    const double scale = limit==0 ? 1.0 : std::min(1.0,double(limit)/(norm+1e-6));
    Require(capture->gradients.size()==(tied ? 1u : 2u),"Shared parameters counted more than once");
    Expect(capture->gradients.at(fw.get()),{float(g1*scale)},"Clipped first gradient");
    if (!tied) Expect(capture->gradients.at(sw.get()),{float(g2*scale)},"Common global scaling factor");
    for (float invalid : {-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        bool rejected=false;
        try { model.SetGradientClipNorm(invalid); }
        catch (const std::invalid_argument&) { rejected=true; }
        Require(rejected,"Invalid gradient clip norm accepted");
    }
#if defined(LAYER_TEST_CPU)
    if (!tied && passes==1 && inputValue==1.0f && limit==1.0f) {
        capture->gradients.clear();
        Write(input.get(),{std::numeric_limits<float>::quiet_NaN()});
        bool rejected=false;
        try { model.Run(input.get(),target.get()); }
        catch (const std::runtime_error&) { rejected=true; }
        Require(rejected,"Non-finite gradients accepted");
        Require(capture->gradients.empty(),"Non-finite gradients reached the optimizer");
    }
#endif
}

int main()
{
    try {
        LOG_INFO() << "Starting global gradient clipping tests";
#if defined(LAYER_TEST_Metal)
        if (!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if (!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        for (bool tied : {false,true}) {
            CheckClipping(tied,1,false,1.0f,1.0f);
            CheckClipping(tied,3,false,1.0f,1.0f);
            // Below threshold distinguishes clipping after averaging from
            // clipping the sum before dividing by the number of passes.
            CheckClipping(tied,3,false,tied ? 0.2f : 0.4f,1.0f);
            CheckClipping(tied,2,true,tied ? 0.2f : 0.4f,1.0f);
            CheckClipping(tied,1,false,0.0f,1.0f);
            CheckClipping(tied,1,false,1.0f,0.0f);
        }
        std::cout << "PASS: global norm, tied gradients, accumulation and partial windows\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
