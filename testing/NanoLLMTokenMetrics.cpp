#include "LayerTestSupport.h"
#include <Examples/NanoLLM/TokenMetrics.h>

using namespace LayerTesting;

int main()
{
    try {
        LOG_INFO()<<"Starting batched NanoLLM token metrics tests";
#if defined(LAYER_TEST_Metal)
        if (!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if (!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        using M=TestMatrix;
        auto& manager=MatrixManager<float,M>::Instance();
        // Nonzero offsets, two distinct samples, ties and out-of-vocabulary IDs.
        auto storage=manager.AllocateMatrix(MatrixShape::Matrix(32),"LogitsStorage");
        auto labels=manager.AllocateMatrix(MatrixShape::Matrix(12),"TargetStorage");
        Values logits(32,-19.0f),targets(12,-19.0f);
        const Values rows={9,1,0, 0,9,1, 0,0,9, 9,9,0, 1,9,0, 0,0,9};
        const Values ids={0,2,2, 0,1,8};
        std::copy(rows.begin(),rows.end(),logits.begin()+3);
        std::copy(ids.begin(),ids.end(),targets.begin()+2);
        Write(storage.get(),logits);Write(labels.get(),targets);
        auto predictions=manager.CreateView(storage,MatrixShape::Sequence(2,3,3),3);
        auto expected=manager.CreateView(labels,MatrixShape::Sequence(2,1,3),2);
        const auto owned=manager.GetAllocatedBytes();
        const auto report=manager.GetString();
        Require(report.find("Total allocated: "+std::to_string(owned)+" bytes")!=std::string::npos,
                "Memory reporting must exclude borrowed view storage");
        const std::vector<char> vocabulary={'a','b','c'};
        for(unsigned repeat=0;repeat<3;++repeat) {
            Require(NanoLLM::DecodeTokenBatch(predictions.get(),vocabulary,0)=="abc","First prediction sample");
            Require(NanoLLM::DecodeTokenBatch(predictions.get(),vocabulary,1)=="abc","Second prediction sample and tie");
            Require(NanoLLM::DecodeTokenBatch(expected.get(),vocabulary,0)=="acc","First target sample");
            Require(NanoLLM::DecodeTokenBatch(expected.get(),vocabulary,1)=="ab?","Second target sample");
            Require(NanoLLM::CountCorrectTokens(predictions.get(),expected.get())==4u,"Accuracy across all samples");
            Require(manager.GetString()==report,"Diagnostic passes must not add or replace matrices");
        }
        bool rejected=false;
        try { NanoLLM::DecodeTokenBatch(predictions.get(),vocabulary,2); }
        catch(const std::invalid_argument&) { rejected=true; }
        Require(rejected,"Invalid decode batch");
        auto mismatch=manager.BatchView(expected,0);
        rejected=false;
        try { NanoLLM::CountCorrectTokens(predictions.get(),mismatch.get()); }
        catch(const std::invalid_argument&) { rejected=true; }
        Require(rejected,"Mismatched accuracy batch");
        Expect(Read(storage.get()),logits,"Diagnostic logits unchanged");
        Expect(Read(labels.get()),targets,"Diagnostic targets unchanged");
        std::cout<<"PASS: batched token metrics without temporary matrix views\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
