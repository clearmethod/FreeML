#pragma once

#include <MatrixLibrary/CPU/MatrixCPU.h>
#if defined(LAYER_TEST_Metal)
#include <MatrixLibrary/GPU/Metal/MatrixMetal_Functions.h>
using TestMatrix = MatrixMetal<float>;
#elif defined(LAYER_TEST_DX11)
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
using TestMatrix = MatrixDX11<float>;
#else
using TestMatrix = MatrixCPU<float>;
#endif
#include <LayerLibrary/Layers.h>
#include <ActivationLibrary/Activations.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace LayerTesting {
using Values = std::vector<float>;
void Require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}
void Near(float actual, float expected, const std::string& where, float tolerance = 3e-4f)
{
    Require(std::isfinite(actual) && std::isfinite(expected) &&
            std::abs(actual - expected) <= tolerance * (1 + std::abs(expected)),
            where + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
template<class M> void Commit(M* mat) { if constexpr (requires { mat->Unmap(); }) mat->Unmap(); }
template<class M> Values Read(M* mat)
{
    Require(mat != nullptr, "Missing matrix");
    const float* ptr = mat->DataRead();
    Values result(ptr, ptr + mat->GetElementCount());
    Commit(mat);
    return result;
}
template<class M> void Write(M* mat, const Values& values)
{
    Require(mat && mat->GetElementCount() == values.size(), "Write size mismatch");
    std::copy(values.begin(), values.end(), mat->DataWrite());
    Commit(mat);
}
Values Pattern(size_t n, float scale = 1.0f)
{
    Values result(n);
    for (size_t i = 0; i < n; ++i) result[i] = scale * (float((i * 7) % 13) - 6.0f) / 7.0f;
    return result;
}
void Expect(const Values& actual, const Values& expected, const std::string& where)
{
    Require(actual.size() == expected.size(), where + ": size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) Near(actual[i], expected[i], where + "[" + std::to_string(i) + "]");
}

template<class M> struct Fixture {
    using Blob = Datablob<float, M>;
    using Ref = typename MatrixManager<float, M>::MatrixRef;
    std::unique_ptr<Layer<float, M>> layer;
    std::unique_ptr<Blob> blob;
    std::vector<Ref> inputs;
    std::string name;
    Ref Make(MatrixShape dims, const Values& values)
    {
        auto ref = MatrixManager<float, M>::Instance().AllocateMatrix(dims, "Test");
        Write(ref.get(), values);
        return ref;
    }
    template<class L> void Use(Blob* data) { layer = std::make_unique<L>(); blob.reset(data); }
    template<class A> void Activation(MatrixShape dims)
    {
        std::string outputName = "ActivationOutput";
        Use<ActivationLayer<float, A, M>>(InitActivationBlob<float, M>(dims, outputName));
    }
    explicit Fixture(const std::string& testName, uint32_t scale = 1u, bool singleBatch = false, bool training = true) : name(testName)
    {
        Require(scale > 0 && scale <= 1024, "Scale must be between 1 and 1024");
        auto checkSize = [](MatrixShape shape) {
            Require(shape.ElementCount() <= UINT32_MAX / sizeof(float),
                    "Scaled matrix exceeds backend buffer size limit");
        };
        MatrixShape dims=MatrixShape::Sequence(singleBatch ? 1u : 2u,3*scale,2*scale);
        checkSize(dims);
        if (name == "ActivationIdentity") Activation<Identity<float>>(dims);
        else if (name == "ActivationRelu") Activation<Relu<float>>(dims);
        else if (name == "ActivationReluOpt") Activation<ReluOpt<float>>(dims);
        else if (name == "ActivationLeakyRelu") Activation<LeakyRelu<float>>(dims);
        else if (name == "ActivationSigmoid") Activation<Sigmoid<float>>(dims);
        else if (name == "ActivationTanh") Activation<Tanh<float>>(dims);
        else if (name == "ActivationGelu") Activation<Gelu<float, M>>(dims);
        else if (name == "Dense" || name == "DenseGelu" || name == "DenseTransposed") {
            dims = MatrixShape::Matrix(3*scale,singleBatch ? 1u : 2*scale);
            auto* data = InitDenseBlob<float, M>(2*scale,3*scale,dims.height,true,false,true,name == "DenseTransposed");
            if (name == "DenseTransposed") Use<Dense<float, Identity<float>, M, true>>(data);
            else if (name == "DenseGelu") Use<Dense<float, Gelu<float,M>, M>>(data);
            else Use<Dense<float, Identity<float>, M>>(data);
        }
        else if (name == "Add" || name == "AddBroadcast") Use<AddLayer<float,M>>(InitAddBlob<float,M>());
        else if (name == "FlattenCopy") Use<FlattenCopyLayer<float,M>>(InitFlattenCopyBlob<float,M>());
        else if (name.starts_with("Dropout"))
            Use<DropoutLayer<float,M>>(InitDropoutBlob<float,M>(name == "DropoutTraining" ? 0.25f : 0.0f, true));
        else if (name == "Embedding") {
            dims = MatrixShape::Sequence(singleBatch ? 1u : 2u,1u,3*scale);
            Use<Embedding<float,M>>(InitEmbeddingBlob<float,M>(4*scale,2*scale,dims.height,dims.batch,true,false));
        }
        else if (name.starts_with("LayerNorm")) Use<LayerNorm<float,M>>(InitLayerNormBlob<float,M>(training,false,name != "LayerNormNoBias"));
        else if (name == "Conv2D") {
            dims = MatrixShape::Image(1u,2,3*scale,3*scale); // Scale spatial dimensions; keep channels fixed.
            checkSize(dims);
            Use<Conv2D<float,Identity<float>,M>>(InitConv2DBlob<float,M>(2,2,MatrixShape::Matrix(2,2),1,1,0,1,true,false,true));
        }
        else if (name == "Conv2DTranspose") {
            dims = MatrixShape::Image(1u,2,2*scale,2*scale);
            checkSize(MatrixShape::Image(1u,2,4*scale,4*scale));
            Use<Conv2DTranspose<float,Identity<float>,M>>(InitConv2DTransposeBlob<float,M>(2,2,MatrixShape::Matrix(2,2),1,2,0,0,1,true,false,true));
        }
        else if (name.starts_with("CausalSelfAttention")) {
            dims = MatrixShape::Matrix((name == "CausalSelfAttentionSixHeads" ? 6u : 4u)*scale,3*scale);
            Use<CausalSelfAttentionLayer<float,Identity<float>,M>>(InitCausalSelfAttentionLayerBlob<float,M>(dims.width,name == "CausalSelfAttentionSixHeads" ? 6u : 2u, name == "CausalSelfAttentionDropout" ? 0.25f : 0.0f,dims.height,true,true));
        }
        else if (name == "TransformerBlock" || name == "TransformerSixHeads") {
            dims = MatrixShape::Matrix((name == "TransformerSixHeads" ? 6u : 2u)*scale,3*scale);
            Use<TransformerBlockLayer<float,M>>(InitTransformerBlockBlob<float,M>(dims.width,name == "TransformerSixHeads" ? 6u : 1u,0,dims.height,1,true,true));
        }
        else if (name == "UNet") {
            dims = MatrixShape::Matrix(2*scale,2*scale);
            checkSize(dims);
            std::vector<ConvSettings> enc(2);
            std::vector<ConvTransposeSettings> dec(2);
            for (auto& s : enc) { s.inChannels=1; s.outChannels=1; }
            for (auto& s : dec) { s.inChannels=1; s.outChannels=1; }
            Use<UNet<float,Identity<float>,M>>(InitUNetBlob<float,M>(enc,dec));
        }
        else if (name == "VariationalAutoencoder") {
            dims = MatrixShape::Matrix(2*scale,2*scale);
            checkSize(dims);
            std::vector<ConvSettings> enc(1); enc[0].inChannels=1; enc[0].outChannels=2;
            checkSize(MatrixShape::Image(1u,2,dims.width,dims.height));
            Use<VariationalAutoencoder<float,Identity<float>,M>>(InitVariationalAutoencoderBlob<float,M>(enc,1,true,0.0f));
        }
        else if (name == "VariationalAutoencoderDecode") {
            dims = MatrixShape::Matrix(2*scale,2*scale);
            checkSize(dims);
            std::vector<ConvTransposeSettings> dec(2);
            for (auto& s : dec) { s.inChannels=1; s.outChannels=1; }
            Use<VariationalAutoencoder_Decode<float,Identity<float>,M>>(InitVariationalAutoencoderDecodeBlob<float,M>(dec));
        }
        else throw std::runtime_error("Unknown layer: " + name);
        auto values = Pattern(dims.CheckedElementCount());
        // Avoid activation kinks when taking finite differences.
        for (auto& v : values) if (v == 0) v = 0.35f;
        if (name == "Embedding") {
            const float ids[] = {1,1,3,0,3,1}; // repeated indices must accumulate
            for (size_t i=0; i<values.size(); ++i) values[i]=ids[i%6]+4.0f*float((i/6)%scale);
        }
        inputs.push_back(Make(dims, values));
        layer->SetInput(blob.get(), inputs[0]);
        if (name == "Add" || name == "AddBroadcast") {
            MatrixShape rhs=dims; if(name == "AddBroadcast") rhs.batch=1u;
            inputs.push_back(Make(rhs, Pattern(rhs.CheckedElementCount(), 0.4f)));
            layer->SetInput(blob.get(), inputs[1], 1);
        }
        layer->EnsureOutputsAllocated(blob.get());
        auto weights = *layer->GetWeights(blob.get());
        const bool convolutional = name == "Conv2D" || name == "Conv2DTranspose" || name == "UNet"
                                   || name == "VariationalAutoencoder" || name == "VariationalAutoencoderDecode";
        const float weightScale = convolutional ? 1.0f : std::sqrt(float(scale));
        for (size_t i = 0; i < weights.size(); ++i) {
            auto vals = Pattern(weights[i]->GetElementCount(), 0.3f);
            for (auto& v : vals) v = (v + float(i+1)*0.015f) / weightScale;
            Write(weights[i].get(), vals);
        }
        if (name == "DropoutInference") blob->Set("TrainingEnabled", 0u);
    }
    void SetBatchSize(uint32_t batches)
    {
        Require(batches>0u,"Batch size must be positive");
        for(size_t index=0;index<inputs.size();++index) {
            auto shape=inputs[index]->GetShape();
            shape.batch=name=="AddBroadcast" && index==1u ? 1u : batches;
            const auto original=Read(inputs[index].get());
            Values values(shape.CheckedElementCount());
            for(size_t i=0;i<values.size();++i) values[i]=original[i%original.size()];
            inputs[index]=Make(shape,values);
            layer->SetInput(blob.get(),inputs[index],uint32_t(index));
        }
        layer->EnsureOutputsAllocated(blob.get());
    }

    void Forward(bool fixedRandom = true)
    {
        layer->Forward(blob.get());
        if (name == "VariationalAutoencoder") {
            auto mu = Read(layer->GetOutput(blob.get(),1).get());
            auto lv = Read(layer->GetOutput(blob.get(),2).get());
            auto eps = Read(blob->GetMatrix("Eps"));
            auto latent = Read(layer->GetOutput(blob.get()).get());
            for (size_t i=0; i<latent.size(); ++i)
                Near(latent[i], mu[i]+std::exp(0.5f*std::clamp(lv[i],-10.0f,4.0f))*eps[i], "VAE sampling");
            if (fixedRandom) {
                const auto sampleNoise=Pattern(eps.size()/blob->GetMatrix("Eps")->GetBatchCount(),0.7f);
                for(size_t i=0;i<eps.size();++i) eps[i]=sampleNoise[i%sampleNoise.size()];
                Write(blob->GetMatrix("Eps"), eps);
                for(size_t i=0;i<mu.size();++i) mu[i] += std::exp(0.5f*std::clamp(lv[i],-10.0f,4.0f))*eps[i];
                Write(layer->GetOutput(blob.get()).get(), mu);
            }
        }
        if (name == "DropoutTraining" && fixedRandom) {
            auto in = Read(inputs[0].get());
            Values mask(in.size()), out(in.size());
            for (size_t i=0;i<in.size();++i) { mask[i]=i%3 ? 1.0f : 0.0f; out[i]=in[i]*mask[i]/0.75f; }
            Write(blob->GetMatrix("Mask"),mask);
            Write(layer->GetOutput(blob.get()).get(),out);
        }
    }
    Values Output() { return Read(layer->GetOutput(blob.get()).get()); }
};

float Activate(const std::string& name, float x)
{
    if (name == "ActivationIdentity") return x;
    if (name == "ActivationRelu" || name == "ActivationReluOpt") return std::max(0.0f,x);
    if (name == "ActivationLeakyRelu") return x>0 ? x : 0.01f*x;
    if (name == "ActivationSigmoid") return 1.0f/(1.0f+std::exp(-x));
    if (name == "ActivationTanh") return std::tanh(x);
    return 0.5f*x*(1+std::erf(x/std::sqrt(2.0f)));
}

// Independent scalar references; these never call matrix operations or layers.
template<class M> Values DenseReference(Datablob<float,M>* blob, const Values& input, size_t inSize, bool gelu=false)
{
    auto* weights=blob->GetMatrix("Dense_Weights");
    auto w=Read(weights);
    bool transposed=blob->GetUInt("Dense_Transposed")!=0;
    size_t outSize=transposed ? weights->GetHeight() : weights->GetWidth();
    auto* bias=blob->GetMatrix("Dense_Bias");
    auto b=bias ? Read(bias) : Values(outSize,0);
    Values result(input.size()/inSize*outSize);
    for(size_t row=0;row<input.size()/inSize;++row) for(size_t col=0;col<outSize;++col) {
        double sum=b[col];
        for(size_t k=0;k<inSize;++k) sum+=double(input[row*inSize+k])*w[transposed ? col*inSize+k : k*outSize+col];
        result[row*outSize+col]=gelu ? Activate("ActivationGelu",float(sum)) : float(sum);
    }
    return result;
}
template<class M> Values NormReference(Datablob<float,M>* blob, const Values& input, size_t width)
{
    auto gamma=Read(blob->GetMatrix("GammaScale"));
    auto* beta=blob->GetMatrix("BetaShift");
    auto b=beta ? Read(beta) : Values(width,0);
    Values result(input.size());
    for(size_t row=0;row<input.size()/width;++row) {
        double mean=0,var=0;
        for(size_t k=0;k<width;++k) mean+=input[row*width+k]/double(width);
        for(size_t k=0;k<width;++k) var+=std::pow(input[row*width+k]-mean,2)/double(width);
        for(size_t k=0;k<width;++k) result[row*width+k]=float((input[row*width+k]-mean)/std::sqrt(var+1e-5))*gamma[k]+b[k];
    }
    return result;
}
template<class M> Values AttentionReference(Datablob<float,M>* blob, const Values& input, size_t width)
{
    auto qkv=DenseReference(blob->GetBlob("CSA_AttentionLayer"),input,width);
    size_t heads=blob->GetUInt("NumHeads"), depth=width/heads, rows=input.size()/width;
    Values mixed(input.size(),0);
    for(size_t row=0;row<rows;++row) for(size_t head=0;head<heads;++head) {
        std::vector<double> scores(row+1);
        for(size_t prev=0;prev<=row;++prev) {
            for(size_t k=0;k<depth;++k)
                scores[prev]+=double(qkv[row*3*width+head*depth+k])*qkv[prev*3*width+width+head*depth+k];
            scores[prev]/=std::sqrt(double(depth));
        }
        double maximum=*std::max_element(scores.begin(),scores.end()), sum=0;
        for(auto& v:scores) { v=std::exp(v-maximum); sum+=v; }
        for(size_t k=0;k<depth;++k) {
            double value=0;
            for(size_t prev=0;prev<=row;++prev) {
                double probability = scores[prev]/sum;
                if (blob->GetFloat("Init_Dropout") > 0) {
                    auto* dropout = blob->GetBlob("CSA_AttentionDropout");
                    auto mask = Read(dropout->GetMatrix("Mask"));
                    probability *= mask[head*rows*rows+row*rows+prev] / (1.0-dropout->GetFloat("Probability"));
                }
                value+=probability*qkv[prev*3*width+2*width+head*depth+k];
            }
            mixed[row*width+head*depth+k]=float(value);
        }
    }
    auto result = DenseReference(blob->GetBlob("CSA_ProjLayer"),mixed,width);
    if (blob->GetFloat("Init_Dropout") > 0) {
        auto* dropout = blob->GetBlob("CSA_ResidualDropout");
        auto mask = Read(dropout->GetMatrix("Mask"));
        for (size_t i=0;i<result.size();++i)
            result[i] *= mask[i] / (1.0f-dropout->GetFloat("Probability"));
    }
    return result;
}
template<class M> Values ConvReference(Datablob<float,M>* blob, const Values& input,
                                      MatrixShape dims, bool transposed=false)
{
    size_t inChannels=blob->GetUInt("InChannels"), outChannels=blob->GetUInt("OutChannels");
    size_t kx=blob->GetUInt("KernelSizeX"), ky=blob->GetUInt("KernelSizeY"), stride=blob->GetUInt("Stride");
    // Fixtures use no padding/dilation. Z is channels, and there is one batch.
    size_t ox=transposed ? (dims.width-1)*stride+kx : (dims.width-kx)/stride+1;
    size_t oy=transposed ? (dims.height-1)*stride+ky : (dims.height-ky)/stride+1;
    auto* bias=blob->GetMatrix("Bias");
    auto b=bias ? Read(bias) : Values(outChannels,0);
    Values result(ox*oy*outChannels);
    for(size_t oc=0;oc<outChannels;++oc) {
        auto w=Read(blob->GetMatrix("KernelWeights_"+std::to_string(oc)));
        for(size_t y=0;y<oy;++y) for(size_t x=0;x<ox;++x) result[oc*ox*oy+y*ox+x]=b[oc];
        for(size_t ic=0;ic<inChannels;++ic) for(size_t y=0;y<(transposed ? dims.height : oy);++y)
        for(size_t x=0;x<(transposed ? dims.width : ox);++x) for(size_t dy=0;dy<ky;++dy) for(size_t dx=0;dx<kx;++dx) {
            size_t src=ic*dims.width*dims.height+(transposed ? y : y*stride+dy)*dims.width+(transposed ? x : x*stride+dx);
            size_t dst=oc*ox*oy+(transposed ? y*stride+dy : y)*ox+(transposed ? x*stride+dx : x);
            result[dst]+=input[src]*w[ic*kx*ky+dy*kx+dx];
        }
    }
    return result;
}

template<class M> void CheckLayerNormCacheLifecycle(const std::string& name)
{
    Fixture<M> training(name);
    for (MatrixShape dims : {MatrixShape::Image(2,2,3,2), MatrixShape::Sequence(1,3,5), MatrixShape::Sequence(3,3,1)}) {
        auto values=Pattern(dims.CheckedElementCount());
        auto input=training.Make(dims,values);
        training.layer->SetInput(training.blob.get(),input);
        training.layer->EnsureOutputsAllocated(training.blob.get());
        training.Forward();
        auto* stats=training.blob->GetMatrix("InvStd");
        Require(stats && stats->GetWidth()==1 && stats->GetHeight()==dims.height && stats->GetPlaneCount()==dims.channels*dims.batch,
                "LayerNorm cache must resize with token and batch counts");
        Values expected(dims.height*dims.channels*dims.batch);
        for(size_t row=0;row<expected.size();++row) {
            double mean=0,var=0;
            for(size_t c=0;c<dims.width;++c) mean+=values[row*dims.width+c]/double(dims.width);
            for(size_t c=0;c<dims.width;++c) var+=std::pow(values[row*dims.width+c]-mean,2)/dims.width;
            expected[row]=float(1/std::sqrt(var+1e-5));
        }
        Expect(Read(stats),expected,"LayerNorm cached invStd");
        // Disabling training must leave even an existing cache untouched.
        Write(stats,Values(expected.size(),-7.0f));
        auto* hat=training.blob->GetMatrix("XHat");
        Write(hat,Values(hat->GetElementCount(),-7.0f));
        training.blob->Set("TrainingEnabled",0u);
        training.layer->EnsureOutputsAllocated(training.blob.get());
        training.Forward();
        Expect(Read(stats),Values(expected.size(),-7.0f),"Inference must not write invStd");
        Expect(Read(hat),Values(hat->GetElementCount(),-7.0f),"Inference must not write XHat");
        training.blob->Set("TrainingEnabled",1u);
        training.Forward();
        Expect(Read(stats),expected,"Training must refresh invStd");
    }
    Fixture<M> inference(name,1u,false,false);
    inference.Forward();
    Require(!inference.blob->GetMatrix("InvStd") && !inference.blob->GetMatrix("XHat"),
            "Inference-only LayerNorm must not allocate training caches");
    auto values=Read(inference.inputs[0].get());
    const auto expected=NormReference(inference.blob.get(),values,3);
    Expect(inference.Output(),expected,"Inference LayerNorm output");
}

template<class M> void CheckForward(Fixture<M>& f)
{
    f.Forward(false);
#ifdef __APPLE__
    if constexpr (std::is_same_v<M, MatrixMetal<float>>)
    {
        if (f.name == "Embedding")
        {
            for (auto* mat : {f.blob->GetMatrix("Embedding_Weights"),
                              f.layer->GetOutput(f.blob.get()).get(), f.inputs[0].get()})
                Require(!mat->m_gInstance->GetMappedPointer(mat->m_dataHandles.m_bufferHandle),
                        "Embedding forward unexpectedly mapped storage to CPU");
        }
        if (f.name == "ActivationIdentity" || f.name == "ActivationGelu")
        {
            auto output = f.layer->GetOutput(f.blob.get());
            Require(!output->m_gInstance->GetMappedPointer(output->m_dataHandles.m_bufferHandle),
                    "Activation forward unexpectedly mapped output to CPU");
        }
    }
#endif
    auto actual = f.Output();
    for (float v : actual) Require(std::isfinite(v), "Nonfinite forward output");
    auto input = Read(f.inputs[0].get());
    Values expected;
    if (f.name.starts_with("Activation")) {
        expected=input; for(auto& x:expected) x=Activate(f.name,x);
    }
    else if (f.name == "FlattenCopy" || f.name == "Dropout" || f.name == "DropoutInference") expected=input;
    else if (f.name == "DropoutTraining") {
        auto mask=Read(f.blob->GetMatrix("Mask")); expected=input;
        for(size_t i=0;i<input.size();++i) {
            Require(mask[i]==0 || mask[i]==1,"Dropout mask must be binary");
            expected[i]*=mask[i]/0.75f;
        }
    }
    else if (f.name == "Add" || f.name == "AddBroadcast") {
        auto rhs=Read(f.inputs[1].get()); expected=input;
        for(size_t i=0;i<input.size();++i) expected[i]+=rhs[i%rhs.size()];
    }
    else if (f.name.starts_with("Dense")) {
        auto w=Read(f.blob->GetMatrix("Dense_Weights")), b=Read(f.blob->GetMatrix("Dense_Bias"));
        expected.resize(4);
        for(size_t row=0;row<2;++row) for(size_t col=0;col<2;++col) {
            float sum=b[col];
            for(size_t k=0;k<3;++k) sum+=input[row*3+k]*w[f.name=="DenseTransposed" ? col*3+k : k*2+col];
            expected[row*2+col]=f.name=="DenseGelu" ? Activate("ActivationGelu",sum) : sum;
        }
    }
    else if (f.name=="Embedding") {
        auto w=Read(f.blob->GetMatrix("Embedding_Weights"));
        for(float token:input) { expected.push_back(w[size_t(token)*2]); expected.push_back(w[size_t(token)*2+1]); }
    }
    else if (f.name.starts_with("LayerNorm")) {
        auto gamma=Read(f.blob->GetMatrix("GammaScale"));
        auto* betaMat=f.blob->GetMatrix("BetaShift");
        auto beta=betaMat ? Read(betaMat) : Values(3,0);
        expected=input;
        for(size_t row=0;row<4;++row) {
            double mean=0,var=0; for(size_t k=0;k<3;++k) mean+=input[row*3+k]/3.0;
            for(size_t k=0;k<3;++k) var+=std::pow(input[row*3+k]-mean,2)/3.0;
            for(size_t k=0;k<3;++k) expected[row*3+k]=float((input[row*3+k]-mean)/std::sqrt(var+1e-5))*gamma[k]+beta[k];
        }
    }
    else if (f.name=="Conv2D" || f.name=="Conv2DTranspose")
        expected=ConvReference(f.blob.get(),input,f.inputs[0]->GetShape(),f.name=="Conv2DTranspose");
    else if (f.name.starts_with("CausalSelfAttention")) expected=AttentionReference(f.blob.get(),input,f.inputs[0]->GetWidth());
    else if (f.name=="TransformerBlock" || f.name=="TransformerSixHeads") {
        const size_t width=f.inputs[0]->GetWidth();
        auto x=AttentionReference(f.blob->GetBlob("AttentionLayerData"),
                                 NormReference(f.blob->GetBlob("LayerNorm0Data"),input,width),width);
        for(size_t i=0;i<x.size();++i) x[i]+=input[i];
        auto mlp=DenseReference(f.blob->GetBlob("DenseFCData"),
                               NormReference(f.blob->GetBlob("LayerNorm1Data"),x,width),width,true);
        mlp=DenseReference(f.blob->GetBlob("DenseProjData"),mlp,4*width);
        expected=x; for(size_t i=0;i<x.size();++i) expected[i]+=mlp[i];
    }
    else if (f.name=="UNet") {
        auto skip=ConvReference(f.blob->GetBlob("EncBlob_0"),input,MatrixShape::Matrix(2,2));
        auto x=ConvReference(f.blob->GetBlob("EncBlob_1"),skip,MatrixShape::Matrix(2,2));
        x=ConvReference(f.blob->GetBlob("DecBlob_0"),x,MatrixShape::Image(1u,1u,2u,2u),true);
        for(size_t i=0;i<x.size();++i) x[i]+=skip[i];
        expected=ConvReference(f.blob->GetBlob("DecBlob_1"),x,MatrixShape::Image(1u,1u,2u,2u),true);
    }
    else if (f.name=="VariationalAutoencoderDecode") {
        auto x=ConvReference(f.blob->GetBlob("ConvTransposeBlob_0"),input,MatrixShape::Image(1u,1u,2u,2u),true);
        expected=ConvReference(f.blob->GetBlob("ConvTransposeBlob_1"),x,MatrixShape::Image(1u,1u,2u,2u),true);
    }
    else if (f.name=="VariationalAutoencoder") {
        auto x=ConvReference(f.blob->GetBlob("ConvBlob_0"),input,MatrixShape::Matrix(2,2));
        auto mu=ConvReference(f.blob->GetBlob("MuBlob"),x,MatrixShape::Image(1u,2u,2u,2u));
        auto lv=ConvReference(f.blob->GetBlob("LogVarBlob"),x,MatrixShape::Image(1u,2u,2u,2u));
        Expect(Read(f.layer->GetOutput(f.blob.get(),1).get()),mu,"VAE mu head");
        Expect(Read(f.layer->GetOutput(f.blob.get(),2).get()),lv,"VAE logvar head");
        auto eps=Read(f.blob->GetMatrix("Eps"));
        expected=mu; for(size_t i=0;i<mu.size();++i) expected[i]+=std::exp(0.5f*lv[i])*eps[i];
    }
    Require(!expected.empty(),"Missing independent forward reference");
    if (!expected.empty()) Expect(actual,expected,"forward");
    // CPU parity verifies backend dispatch and composite layer wiring.
    if (f.name != "CausalSelfAttentionDropout") {
        Fixture<MatrixCPU<float>> reference(f.name);
        reference.Forward();
        f.Forward();
        Expect(f.Output(),reference.Output(),"CPU parity");
    }
    auto out=f.layer->GetOutput(f.blob.get());
    Require(out && out->GetElementCount()>0,"Empty output");
    if (f.name=="FlattenCopy") Require(out->GetShape()==MatrixShape::Sequence(2,6,1),"Flatten dimensions");
    if ((f.name.starts_with("CausalSelfAttention") && f.name != "CausalSelfAttentionDropout") || f.name=="TransformerBlock" || f.name=="TransformerSixHeads") {
        auto before=f.Output(), modified=input;
        for(size_t i=2*f.inputs[0]->GetWidth();i<3*f.inputs[0]->GetWidth();++i) modified[i]+=2.0f;
        Write(f.inputs[0].get(),modified); f.Forward(); auto after=f.Output();
        for(size_t i=0;i<2*f.inputs[0]->GetWidth();++i) Near(after[i],before[i],"Causal prefix");
        Write(f.inputs[0].get(),input);
    }
    if (f.name == "CausalSelfAttention" || f.name == "CausalSelfAttentionSixHeads") {
        auto originalInput = f.inputs[0];
        const uint32_t originalHeads = f.blob->GetUInt("NumHeads");
        const uint32_t width = originalInput->GetWidth();
        // Exercise both shape resizing and head-storage rebinding on one live layer.
        for (uint32_t tokens : {5u, 2u, 3u}) {
            f.blob->Set("NumHeads", tokens == 2 ? (width == 6 ? 3u : 1u) : originalHeads);
            auto resized = f.Make(MatrixShape::Matrix(width,tokens), Pattern(width*tokens));
            f.layer->SetInput(f.blob.get(),resized);
            f.layer->EnsureOutputsAllocated(f.blob.get());
            f.Forward();
            Expect(f.Output(),AttentionReference(f.blob.get(),Read(resized.get()),width),
                   "Attention shared-storage resize");
        }
        f.blob->Set("NumHeads",originalHeads);
        f.layer->SetInput(f.blob.get(),originalInput);
        f.layer->EnsureOutputsAllocated(f.blob.get());
    }

}

template<class M> void CheckBackward(Fixture<M>& f)
{
    f.Forward();
    auto out=f.layer->GetOutput(f.blob.get());
    auto upstream=Pattern(out->GetElementCount(),0.8f);
    auto err=f.Make(out->GetShape(),upstream);
    f.layer->SetErrorInput(f.blob.get(),err);
    Values linearBefore;
    if(f.name=="DenseGelu") linearBefore=Read(f.blob->GetMatrix("Dense_LinearOut"));
    f.layer->Backwards(f.blob.get());
    if(f.name=="DenseGelu")
        Expect(Read(f.blob->GetMatrix("Dense_LinearOut")),linearBefore,
               "Fused GELU backward must not overwrite forward scratch with a derivative");
#ifdef __APPLE__
    if constexpr (std::is_same_v<M, MatrixMetal<float>>)
    {
        if (f.name == "Embedding")
        {
            auto gradient = f.blob->GetMatrix("Embedding_WUpdate");
            Require(!gradient->m_gInstance->GetMappedPointer(gradient->m_dataHandles.m_bufferHandle),
                    "Embedding backward unexpectedly mapped gradient to CPU");
        }
        if (f.name == "ActivationIdentity")
        {
            auto gradient = f.layer->GetOutputError(f.blob.get());
            Require(!gradient->m_gInstance->GetMappedPointer(gradient->m_dataHandles.m_bufferHandle),
                    "Identity backward unexpectedly mapped output to CPU");
        }
    }
#endif
    auto weights=*f.layer->GetWeights(f.blob.get());
    auto grads=*f.layer->GetGradients(f.blob.get());
    Require(weights.size()==grads.size(),"Weight/gradient count mismatch");
    std::vector<Values> savedGrads;
    for(size_t i=0;i<grads.size();++i) {
        Require(weights[i]->GetElementCount()==grads[i]->GetElementCount(),"Gradient dimensions");
        savedGrads.push_back(Read(grads[i].get()));
    }
    std::vector<Values> inputGrads;
    for(size_t i=0;i<f.inputs.size();++i) inputGrads.push_back(Read(f.layer->GetOutputError(f.blob.get(),uint32_t(i)).get()));
    auto objective=[&]() {
        Values values;
        if (f.name == "CausalSelfAttentionDropout")
            values = AttentionReference(f.blob.get(),Read(f.inputs[0].get()),f.inputs[0]->GetWidth());
        else { f.Forward(); values=f.Output(); }
        double sum=0;
        for(size_t i=0;i<values.size();++i) sum+=double(values[i])*upstream[i];
        return sum;
    };
    auto gradientCheck=[&](M* variable,const Values& analytic,const std::string& label) {
        auto values=Read(variable);
        Require(values.size()==analytic.size(),label+": size mismatch");
        constexpr float h=0.002f;
        for(size_t i=0;i<values.size();++i) {
            const float original=values[i];
            values[i]=original+h; Write(variable,values); double plus=objective();
            values[i]=original-h; Write(variable,values); double minus=objective();
            values[i]=original; Write(variable,values);
            const float numerical=float((plus-minus)/(2*h));
            Require(std::isfinite(analytic[i]) && std::abs(analytic[i]-numerical)<=0.0007f+0.015f*std::abs(numerical),
                label+"["+std::to_string(i)+"] expected "+std::to_string(numerical)+", got "+std::to_string(analytic[i]));
        }
    };
    if (f.name=="Embedding") {
        for(float value:inputGrads[0]) Near(value,0,"Token IDs have no gradient");
    }
    else for(size_t i=0;i<f.inputs.size();++i) gradientCheck(f.inputs[i].get(),inputGrads[i],"input "+std::to_string(i));
    for(size_t i=0;i<weights.size();++i) gradientCheck(weights[i].get(),savedGrads[i],"weight "+std::to_string(i));
    // A repeated backward pass must overwrite gradients, not accumulate stale state.
    if (f.name != "CausalSelfAttentionDropout") f.Forward();
    f.layer->Backwards(f.blob.get());
    grads=*f.layer->GetGradients(f.blob.get());
    for(size_t i=0;i<grads.size();++i) Expect(Read(grads[i].get()),savedGrads[i],"repeated backward");
}
const std::vector<std::string> cases={
#define LAYER_CASE(name) #name,
#include "LayerCases.def"
#undef LAYER_CASE
};
}
