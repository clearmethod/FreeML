#include "LayerTestSupport.h"

using namespace LayerTesting;
using M=TestMatrix;

void CheckBatch(const std::string& name,unsigned batches)
{
    Fixture<M> batched(name);
    const auto base=batched.inputs[0]->GetShape();
    const auto shape=MatrixShape::Image(batches,base.channels,base.width,base.height);
    Values input=Pattern(shape.CheckedElementCount(),0.7f);
    if(name=="Embedding") for(unsigned i=0;i<input.size();++i) input[i]=float((i*3+i/2)%4);
    auto data=batched.Make(shape,input);batched.inputs[0]=data;
    batched.layer->SetInput(batched.blob.get(),data);
    Values rhs;
    if(name=="Add" || name=="AddBroadcast") {
        const auto rhsShape=MatrixShape::Sequence(name=="AddBroadcast"?1u:batches,base.width,base.height);
        rhs=Pattern(rhsShape.CheckedElementCount(),0.3f);
        auto other=batched.Make(rhsShape,rhs);batched.inputs[1]=other;
        batched.layer->SetInput(batched.blob.get(),other,1u);
    }
    batched.layer->EnsureOutputsAllocated(batched.blob.get());batched.Forward();
    auto output=batched.layer->GetOutput(batched.blob.get());
    Require(output->GetBatchCount()==batches,name+": output lost batch dimension");
    Values actual=Read(output.get()),upstream=Pattern(output->GetElementCount(),0.8f);
    auto error=batched.Make(output->GetShape(),upstream);
    batched.layer->SetErrorInput(batched.blob.get(),error);batched.layer->Backwards(batched.blob.get());
    Values inputGradient=Read(batched.layer->GetOutputError(batched.blob.get()).get());
    std::vector<Values> parameterGradients;
    auto weights=*batched.layer->GetWeights(batched.blob.get());
    auto gradients=*batched.layer->GetGradients(batched.blob.get());
    for(auto& gradient:gradients) parameterGradients.emplace_back(gradient->GetElementCount(),0);
    const unsigned sampleInput=base.width*base.height*base.channels,sampleOutput=unsigned(actual.size())/batches;
    for(unsigned batch=0;batch<batches;++batch) {
        Fixture<M> single(name,1u,true);
        auto singleWeights=*single.layer->GetWeights(single.blob.get());
        Require(singleWeights.size()==weights.size(),name+": parameter count changed");
        for(unsigned p=0;p<weights.size();++p) Write(singleWeights[p].get(),Read(weights[p].get()));
        auto sample=single.Make(MatrixShape::Image(1,base.channels,base.width,base.height),
            Values(input.begin()+batch*sampleInput,input.begin()+(batch+1)*sampleInput));
        single.inputs[0]=sample;single.layer->SetInput(single.blob.get(),sample);
        if(!rhs.empty()) {
            const unsigned offset=name=="AddBroadcast"?0u:batch*sampleInput;
            auto other=single.Make(sample->GetShape(),Values(rhs.begin()+offset,rhs.begin()+offset+sampleInput));
            single.inputs[1]=other;single.layer->SetInput(single.blob.get(),other,1u);
        }
        single.layer->EnsureOutputsAllocated(single.blob.get());single.Forward();
        Expect(single.Output(),Values(actual.begin()+batch*sampleOutput,actual.begin()+(batch+1)*sampleOutput),name+": sample output");
        auto sampleError=single.Make(single.layer->GetOutput(single.blob.get())->GetShape(),
            Values(upstream.begin()+batch*sampleOutput,upstream.begin()+(batch+1)*sampleOutput));
        single.layer->SetErrorInput(single.blob.get(),sampleError);single.layer->Backwards(single.blob.get());
        Expect(Read(single.layer->GetOutputError(single.blob.get()).get()),
            Values(inputGradient.begin()+batch*sampleInput,inputGradient.begin()+(batch+1)*sampleInput),name+": input gradient");
        auto sampleGradients=*single.layer->GetGradients(single.blob.get());
        for(unsigned p=0;p<sampleGradients.size();++p) {
            auto values=Read(sampleGradients[p].get());
            for(unsigned i=0;i<values.size();++i) parameterGradients[p][i]+=values[i];
        }
    }
    for(unsigned p=0;p<gradients.size();++p)
        Expect(Read(gradients[p].get()),parameterGradients[p],name+": summed parameter gradient");
    std::cout<<"PASS: "<<name<<" batch="<<batches<<'\n';
}

int main()
{
    try {
        LOG_INFO()<<"Starting physical batching regression tests";
#if defined(LAYER_TEST_Metal)
        if(!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if(!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        for(const char* name:{"Conv2D","Conv2DTranspose"}) {
            Fixture<M> invalid(name);
            if(invalid.name=="Conv2D") invalid.blob->Set("KernelSizeX",UINT32_MAX);
            else invalid.blob->Set("Padding",UINT32_MAX);
            bool rejected=false;
            try { invalid.layer->EnsureOutputsAllocated(invalid.blob.get()); }
            catch(const std::invalid_argument&) { rejected=true; }
            Require(rejected,"Invalid convolution geometry must be rejected before allocation");
        }
        for(const char* name : {"ActivationIdentity","ActivationRelu","ActivationReluOpt","ActivationLeakyRelu",
            "ActivationSigmoid","ActivationTanh","ActivationGelu","Add","AddBroadcast","Dropout",
            "Embedding","LayerNorm","LayerNormNoBias","FlattenCopy","Dense","DenseGelu","DenseTransposed",
            "CausalSelfAttention","CausalSelfAttentionSixHeads","TransformerBlock","TransformerSixHeads",
            "Conv2D","Conv2DTranspose","UNet","VariationalAutoencoder","VariationalAutoencoderDecode"})
            for(unsigned batches : {1u,2u,3u}) CheckBatch(name,batches);
        return 0;
    } catch(const std::exception& error) {std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
}
