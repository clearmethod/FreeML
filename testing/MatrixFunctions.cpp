#include <MatrixLibrary/CPU/MatrixCPU.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/MatrixManager.h>
#include <ActivationLibrary/Gelu.h>
#include <LossLibrary/CategoricalCrossEntropyWithLogits.h>

#if defined(MATRIX_TEST_Metal)
#include <MatrixLibrary/GPU/Metal/MatrixMetal_Functions.h>
using Mat = MatrixMetal<float>;
#elif defined(MATRIX_TEST_DX11)
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11_Functions.h>
using Mat = MatrixDX11<float>;
#else
using Mat = MatrixCPU<float>;
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Values = std::vector<float>;
using Base = MatrixBase<float>;

void Commit(Mat* mat)
{
#if defined(MATRIX_TEST_Metal) || defined(MATRIX_TEST_DX11)
    mat->Unmap();
#else
    (void)mat;
#endif
}

struct DeleteMatrix {
    void operator()(Mat* mat) const
    {
        Commit(mat);
#if defined(MATRIX_TEST_CPU)
        mat->FreeMemory();
#endif
        delete mat;
    }
};
using Ref = std::unique_ptr<Mat, DeleteMatrix>;

void Write(Mat* mat, const Values& values)
{
    if (values.size() != mat->GetElementCount())
        throw std::runtime_error("Fixture size does not match matrix dimensions");
    std::copy(values.begin(), values.end(), mat->DataWrite());
    Commit(mat);
}

Ref Make(MatrixShape dims, const Values& values = {})
{
    Ref mat(new Mat);
    mat->SetShape(dims);
    if (!mat->AllocateMemory(mat->GetElementCount()))
        throw std::runtime_error("Matrix allocation failed");
    // A shader that silently fails to dispatch must never look like a passing test.
    Write(mat.get(), values.empty() ? Values(mat->GetElementCount(),
          std::numeric_limits<float>::quiet_NaN()) : values);
    return mat;
}

Values Read(Mat* mat)
{
    const float* ptr = mat->DataRead();
    Values values(ptr, ptr + mat->GetElementCount());
    Commit(mat);
    return values;
}

void Near(float actual, float expected, float tolerance = 2e-4f)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * (1 + std::abs(expected)))
        throw std::runtime_error("expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

void Expect(Mat* mat, const Values& expected, float tolerance = 2e-4f)
{
    const Values actual = Read(mat);
    if (actual.size() != expected.size()) throw std::runtime_error("Result size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        try { Near(actual[i], expected[i], tolerance); }
        catch (const std::exception& error) {
            throw std::runtime_error("element " + std::to_string(i) + ": " + error.what());
        }
    }
}

using Test = std::pair<std::string, void(*)()>;
std::vector<Test>& Tests() { static std::vector<Test> tests; return tests; }
struct Register { Register(const char* name, void(*fn)()) { Tests().emplace_back(name, fn); } };
#define MATRIX_TEST(name) void Test_##name(); Register register_##name(#name, Test_##name); void Test_##name()

MATRIX_TEST(FourDimensionalReductions)
{
    const auto shape=MatrixShape::Image(3,2,7,5);
    Values values(shape.CheckedElementCount());
    for(unsigned i=0;i<values.size();++i) values[i]=float(int(i%19)-9)*0.25f;
    Values backing(280,-13);std::copy(values.begin(),values.end(),backing.begin()+70);
    auto input=Make(MatrixShape::Image(4,2,7,5),backing);
    Mat view;input->GetPlaneView(&view,2);view.SetShape(shape);
    for(uint32_t bits=0;bits<16;++bits) {
        const auto axes=static_cast<MatrixAxes>(bits);
        const auto resultShape=shape.Reduced(axes);
        Values expected(resultShape.CheckedElementCount(),0);
        for(unsigned b=0;b<3;++b) for(unsigned c=0;c<2;++c)
        for(unsigned y=0;y<5;++y) for(unsigned x=0;x<7;++x)
            expected[resultShape.Index((bits&1u)?0:b,(bits&2u)?0:c,(bits&4u)?0:x,(bits&8u)?0:y)]
                +=values[shape.Index(b,c,x,y)];
        auto out=Make(MatrixShape::Matrix(unsigned(expected.size())+8),Values(expected.size()+8,-17));
        Mat outputView;outputView.SetData(out->GetData(),out->GetOffset()+4);outputView.SetShape(resultShape);
        ReduceSum(&outputView,&view,axes);
        Values outputBacking(expected.size()+8,-17);std::copy(expected.begin(),expected.end(),outputBacking.begin()+4);
        Expect(out.get(),outputBacking);
    }
    Expect(input.get(),backing);
}
MATRIX_TEST(FourDimensionalQueuedReduction)
{
    const auto shape=MatrixShape::Image(3,2,7,5);
    auto input=Make(shape),out=Make(shape.Reduced(MatrixAxes::Batch|MatrixAxes::Height));
    auto consumed=Make(out->GetShape());
#if defined(MATRIX_TEST_Metal)
    auto* gpu=input->m_gInstance;gpu->WaitForGPU();Obj cb=gpu->GetCommandBuffer();
#endif
    Fill(input.get(),2.0f);
    ReduceSum(out.get(),input.get(),MatrixAxes::Batch|MatrixAxes::Height);
    Scale(consumed.get(),out.get(),2.0f);
#if defined(MATRIX_TEST_Metal)
    if(gpu->GetCommandBuffer()!=cb || gpu->m_lastSubmitted)
        throw std::runtime_error("Four-dimensional reduction submitted GPU work");
    MetalManager::CBufferParams params={};
    gpu->GetCBufferParams(input->m_dataHandles.m_cbufferHandle,&params);
    if(params.channels!=2u || params.batch!=3u || params.z!=6u)
        throw std::runtime_error("GPU metadata lost explicit batch/channel dimensions");
#endif
    Expect(consumed.get(),Values(consumed->GetElementCount(),60.0f));
}
MATRIX_TEST(FourDimensionalBroadcast)
{
    const auto shape=MatrixShape::Image(3,2,7,5);
    for(uint32_t axes=0;axes<16;++axes) {
        const auto leftShape=shape.Reduced(static_cast<MatrixAxes>(axes));
        const auto rightShape=shape.Reduced(static_cast<MatrixAxes>(15u^axes));
        Values a(leftShape.CheckedElementCount()),b(rightShape.CheckedElementCount());
        for(unsigned i=0;i<a.size();++i) a[i]=float(int(i%13)-6)*0.25f;
        for(unsigned i=0;i<b.size();++i) b[i]=float(int(i%7)-3)*0.5f;
        auto left=Make(leftShape,a),right=Make(rightShape,b),out=Make(shape);
        for(uint32_t operation=0;operation<3;++operation) {
            Values expected(shape.CheckedElementCount());
            for(unsigned batch=0;batch<3;++batch) for(unsigned c=0;c<2;++c)
            for(unsigned y=0;y<5;++y) for(unsigned x=0;x<7;++x) {
                const float lhs=a[leftShape.Index((axes&1u)?0:batch,(axes&2u)?0:c,(axes&4u)?0:x,(axes&8u)?0:y)];
                const float rhs=b[rightShape.Index((axes&1u)?batch:0,(axes&2u)?c:0,(axes&4u)?x:0,(axes&8u)?y:0)];
                expected[shape.Index(batch,c,x,y)]=operation==0?lhs+rhs:operation==1?lhs-rhs:lhs*rhs;
            }
            if(operation==0) Add(out.get(),left.get(),right.get());
            else if(operation==1) Sub(out.get(),left.get(),right.get());
            else BroadcastBinary(out.get(),left.get(),right.get(),operation);
            Expect(out.get(),expected);
        }
        Expect(left.get(),a);Expect(right.get(),b);
    }
    auto out=Make(shape,Values(shape.CheckedElementCount(),2)),bias=Make(MatrixShape::Matrix(7),Values(7,3));
    Add(out.get(),out.get(),bias.get());Expect(out.get(),Values(out->GetElementCount(),5));
    Sub(out.get(),bias.get(),out.get());Expect(out.get(),Values(out->GetElementCount(),-2));
}
MATRIX_TEST(FourDimensionalConcat)
{
    for(auto axis : {MatrixAxis::Batch,MatrixAxis::Channels,MatrixAxis::Width,MatrixAxis::Height}) {
        auto a=MatrixShape::Image(2,2,3,5),b=a;
        if(axis==MatrixAxis::Batch) b.batch=3;
        if(axis==MatrixAxis::Channels) b.channels=3;
        if(axis==MatrixAxis::Width) b.width=2;
        if(axis==MatrixAxis::Height) b.height=2;
        const auto combined=a.Concatenated(b,axis);
        auto left=Make(a),right=Make(b),out=Make(combined);
#if defined(MATRIX_TEST_Metal)
        auto* gpu=out->m_gInstance;gpu->WaitForGPU();Obj cb=gpu->GetCommandBuffer();
#endif
        Fill(left.get(),2.0f);Fill(right.get(),3.0f);
        Concat(out.get(),left.get(),right.get(),axis);
#if defined(MATRIX_TEST_Metal)
        if(gpu->GetCommandBuffer()!=cb || gpu->m_lastSubmitted)
            throw std::runtime_error("Concatenation submitted GPU work");
#endif
        Values expected(combined.CheckedElementCount());
        for(unsigned batch=0;batch<combined.batch;++batch) for(unsigned c=0;c<combined.channels;++c)
        for(unsigned y=0;y<combined.height;++y) for(unsigned x=0;x<combined.width;++x) {
            const bool first=axis==MatrixAxis::Batch?batch<a.batch:axis==MatrixAxis::Channels?c<a.channels:
                axis==MatrixAxis::Width?x<a.width:y<a.height;
            expected[combined.Index(batch,c,x,y)]=first?2.0f:3.0f;
        }
        Expect(out.get(),expected);
    }
}

template<TransposeMode Mode> void FourDimensionalProducts()
{
    constexpr bool tl=Mode==TransposeMode::Left || Mode==TransposeMode::Both;
    constexpr bool tr=Mode==TransposeMode::Right || Mode==TransposeMode::Both;
    for(auto size : {MatrixShape::Image(1,7,5,3),MatrixShape::Image(1,23,19,17),MatrixShape::Image(1,193,97,65)})
    for(unsigned leftChannels : {1u,2u}) {
        const unsigned m=size.height,n=size.width,k=size.channels;
        const auto a=MatrixShape::Image(3,leftChannels,tl?m:k,tl?k:m);
        const auto b=MatrixShape::Image(1,2,tr?k:n,tr?n:k);
        const auto o=MatrixShape::Image(3,2,n,m);
        auto strides=[](const MatrixShape& shape) {
            const unsigned row=shape.width*2*shape.channels+5;
            return MatrixStrides{row,2u,shape.width*2,row*shape.height+17};
        };
        const auto as=strides(a),bs=strides(b),os=strides(o);
        auto count=[](const MatrixShape& shape,MatrixStrides st) {
            return 3u+(shape.batch-1)*st.batch+(shape.channels-1)*st.channel
                +(shape.height-1)*st.row+(shape.width-1)*st.column+9;
        };
        Values av(count(a,as)),bv(count(b,bs)),expected(count(o,os),-19);
        for(unsigned i=0;i<av.size();++i) av[i]=float(int(i%17)-8)*0.125f;
        for(unsigned i=0;i<bv.size();++i) bv[i]=float(int(i%13)-6)*0.125f;
        for(unsigned batch=0;batch<3;++batch) for(unsigned c=0;c<2;++c)
        for(unsigned y=0;y<m;++y) for(unsigned x=0;x<n;++x) {
            double sum=0;
            for(unsigned inner=0;inner<k;++inner)
                sum+=double(av[3+batch*as.batch+(leftChannels==1?0:c)*as.channel+(tl?inner:y)*as.row+(tl?y:inner)*as.column])
                    *bv[3+c*bs.channel+(tr?x:inner)*bs.row+(tr?inner:x)*bs.column];
            expected[3+batch*os.batch+c*os.channel+y*os.row+x*os.column]=float(sum);
        }
        auto left=Make(MatrixShape::Matrix(unsigned(av.size())),av);
        auto right=Make(MatrixShape::Matrix(unsigned(bv.size())),bv);
        auto output=Make(MatrixShape::Matrix(unsigned(expected.size())),Values(expected.size(),-19));
        Mat va,vb,vo;
        va.SetData(left->GetData(),3);va.SetShape(a);
        vb.SetData(right->GetData(),3);vb.SetShape(b);
        vo.SetData(output->GetData(),3);vo.SetShape(o);
        MatMul_Batched<Mode>(&vo,&va,&vb,os,as,bs);
        Expect(output.get(),expected);Expect(left.get(),av);Expect(right.get(),bv);
    }
}
MATRIX_TEST(FourDimensionalMatMul) { FourDimensionalProducts<TransposeMode::None>(); }
MATRIX_TEST(FourDimensionalMatMulLeft) { FourDimensionalProducts<TransposeMode::Left>(); }
MATRIX_TEST(FourDimensionalMatMulRight) { FourDimensionalProducts<TransposeMode::Right>(); }
MATRIX_TEST(FourDimensionalMatMulBoth) { FourDimensionalProducts<TransposeMode::Both>(); }
MATRIX_TEST(FourDimensionalConvolution)
{
    for(bool transposed : {false,true}) for(unsigned stride : {1u,2u}) for(unsigned dilation : {1u,2u}) {
        constexpr unsigned batches=3,inChannels=2,outChannels=2,w=5,h=4,kw=3,kh=2,padding=1;
        const unsigned outputPadding=transposed && stride==2u ? 1u:0u;
        const unsigned ow=transposed?(w-1)*stride-2*padding+dilation*(kw-1)+outputPadding+1
            :(w+2*padding-dilation*(kw-1)-1)/stride+1;
        const unsigned oh=transposed?(h-1)*stride-2*padding+dilation*(kh-1)+outputPadding+1
            :(h+2*padding-dilation*(kh-1)-1)/stride+1;
        const auto is=MatrixShape::Image(batches,inChannels,w,h),os=MatrixShape::Image(batches,outChannels,ow,oh);
        const auto ks=MatrixShape::Image(1u,inChannels,kw,kh);
        Values values(is.CheckedElementCount()),error(os.CheckedElementCount()),expectedOutput(error.size()),expectedDX(values.size(),0);
        std::vector<Values> weights(outChannels,Values(ks.CheckedElementCount())),expectedDW(outChannels,Values(ks.CheckedElementCount(),0));
        for(unsigned i=0;i<values.size();++i) values[i]=float(int(i%13)-6)*0.125f;
        for(unsigned i=0;i<error.size();++i) error[i]=float(int(i%17)-8)*0.125f;
        for(unsigned oc=0;oc<outChannels;++oc) for(unsigned i=0;i<weights[oc].size();++i)
            weights[oc][i]=float(int((i+oc*3)%11)-5)*0.125f;
        const Values bias{0.25f,-0.5f};Values expectedDB(outChannels,0);
        for(unsigned b=0;b<batches;++b) for(unsigned oc=0;oc<outChannels;++oc)
            for(unsigned y=0;y<oh;++y) for(unsigned x=0;x<ow;++x) {
                expectedOutput[os.Index(b,oc,x,y)]=bias[oc];expectedDB[oc]+=error[os.Index(b,oc,x,y)];
            }
        // Scatter from input pixels, independently of the implementation's output gather.
        for(unsigned b=0;b<batches;++b) for(unsigned ic=0;ic<inChannels;++ic)
        for(unsigned iy=0;iy<h;++iy) for(unsigned ix=0;ix<w;++ix)
        for(unsigned oc=0;oc<outChannels;++oc) for(unsigned ky=0;ky<kh;++ky) for(unsigned kx=0;kx<kw;++kx) {
            int x=transposed?int(ix*stride+kx*dilation)-int(padding):int(ix+padding)-int(kx*dilation);
            int y=transposed?int(iy*stride+ky*dilation)-int(padding):int(iy+padding)-int(ky*dilation);
            if(x<0 || y<0 || (!transposed && (x%int(stride) || y%int(stride)))) continue;
            if(!transposed) {x/=int(stride);y/=int(stride);}
            if(x>=int(ow) || y>=int(oh)) continue;
            const auto inputIndex=is.Index(b,ic,ix,iy),weightIndex=ks.Index(0,ic,kx,ky),outputIndex=os.Index(b,oc,unsigned(x),unsigned(y));
            expectedOutput[outputIndex]+=values[inputIndex]*weights[oc][weightIndex];
            expectedDX[inputIndex]+=weights[oc][weightIndex]*error[outputIndex];
            expectedDW[oc][weightIndex]+=values[inputIndex]*error[outputIndex];
        }
        auto input=Make(is,values),upstream=Make(os,error),out=Make(os),dx=Make(is),biasMatrix=Make(MatrixShape::Matrix(2),bias);
        auto db=Make(os.Reduced(MatrixAxes::Batch|MatrixAxes::Width|MatrixAxes::Height));
        std::vector<Ref> kernels,gradients;
        for(unsigned oc=0;oc<outChannels;++oc) {kernels.push_back(Make(ks,weights[oc]));gradients.push_back(Make(ks));}
#if defined(MATRIX_TEST_Metal)
        auto* gpu=input->m_gInstance;gpu->WaitForGPU();Obj cb=gpu->GetCommandBuffer();
#endif
        Clear(dx.get());
        for(unsigned oc=0;oc<outChannels;++oc) {
            Clear(gradients[oc].get());
            if(transposed) {
                Conv2DTransposeSingleChannel(out.get(),input.get(),kernels[oc].get(),biasMatrix.get(),oc,stride,dilation,padding);
                Conv2DTransposeSingleChannelBackwards(upstream.get(),input.get(),kernels[oc].get(),gradients[oc].get(),dx.get(),oc,stride,dilation,padding);
            } else {
                Conv2DSingleChannel(out.get(),input.get(),kernels[oc].get(),biasMatrix.get(),oc,stride,dilation,padding);
                Conv2DSingleChannelBackwards(upstream.get(),input.get(),kernels[oc].get(),gradients[oc].get(),dx.get(),oc,stride,dilation,padding);
            }
        }
        ReduceSum(db.get(),upstream.get(),MatrixAxes::Batch|MatrixAxes::Width|MatrixAxes::Height);
#if defined(MATRIX_TEST_Metal)
        if(gpu->GetCommandBuffer()!=cb || gpu->m_lastSubmitted)
            throw std::runtime_error("Batched convolution synchronized GPU work");
        if(gpu->GetMappedPointer(input->m_dataHandles.m_bufferHandle))
            throw std::runtime_error("Batched convolution downloaded its input");
#endif
        Expect(out.get(),expectedOutput);Expect(dx.get(),expectedDX);Expect(db.get(),expectedDB);
        for(unsigned oc=0;oc<outChannels;++oc) Expect(gradients[oc].get(),expectedDW[oc]);
    }
}

MATRIX_TEST(Copy)
{
    auto in = Make(MatrixShape::Image(1u,2,3,2), {1,2,3,4,5,6,7,8,9,10,11,12}), out = Make(MatrixShape::Image(1u,2,3,2));
    Copy(out.get(), in.get()); Expect(out.get(), Read(in.get()));
}
MATRIX_TEST(CopyRange)
{
    auto in = Make(MatrixShape::Matrix(6), {1,2,3,4,5,6}), out = Make(MatrixShape::Matrix(7), {-1,-1,-1,-1,-1,-1,-1});
    CopyRange(out.get(), in.get(), 2, 1, 3); Expect(out.get(), {-1,-1,2,3,4,-1,-1});
    CopyRange(out.get(), in.get(), 7, 6, 0); Expect(out.get(), {-1,-1,2,3,4,-1,-1});
}
MATRIX_TEST(Clear)
{
    auto out = Make(MatrixShape::Image(1u,2,3,2)); Clear(out.get()); Expect(out.get(), Values(12,0));
}
MATRIX_TEST(ClearSlice)
{
    auto parent = Make(MatrixShape::Image(1u,3,2,2), {1,2,3,4,5,6,7,8,9,10,11,12});
    Mat slice; parent->GetPlaneView(&slice, 1);
    // Dirty CPU edits outside the cleared view must survive.
    parent->DataWrite()[0] = 20.0f;
    Clear(&slice);
    Expect(parent.get(), {20,2,3,4,0,0,0,0,9,10,11,12});
}
MATRIX_TEST(ClearMappedBuffer)
{
    auto mat = Make(MatrixShape::Matrix(4), {1,2,3,4});
    mat->DataWrite()[0] = 99.0f;
    Clear(mat.get());
#if defined(MATRIX_TEST_Metal)
    auto* manager = mat->m_gInstance;
    if (!manager->m_workToCommit || manager->m_lastSubmitted)
        throw std::runtime_error("Clear unexpectedly submitted GPU work");
    if (manager->GetMappedPointer(mat->m_dataHandles.m_bufferHandle))
        throw std::runtime_error("Clear retained stale CPU mapping");
#endif
    Expect(mat.get(), Values(4,0.0f));
}
MATRIX_TEST(FillScalar)
{
    auto out = Make(MatrixShape::Image(1u,2,3,2)); Fill(out.get(), -2.5f); Expect(out.get(), Values(12,-2.5f));
}
MATRIX_TEST(FillMatrix)
{
    auto out = Make(MatrixShape::Image(1u,2,3,2)), val = Make(MatrixShape::Matrix(1), {3.5f});
    Fill(out.get(), val.get()); Expect(out.get(), Values(12,3.5f));
}

void Binary(const std::function<void(Mat*,Mat*,Mat*)>& fn, const Values& expected)
{
    const Values a = {1,-2,3,4,0,-6,7,8}, b = {2,3,-4,5,6,7,-8,9};
    auto l = Make(MatrixShape::Image(1u,2,2,2), a), r = Make(MatrixShape::Image(1u,2,2,2), b), out = Make(MatrixShape::Image(1u,2,2,2));
    fn(out.get(), l.get(), r.get()); Expect(out.get(), expected);
    fn(l.get(), l.get(), r.get()); Expect(l.get(), expected);
    Write(l.get(), a); Write(r.get(), b);
    fn(r.get(), l.get(), r.get()); Expect(r.get(), expected);
}
MATRIX_TEST(Add) { Binary([](auto o,auto l,auto r){Add(o,l,r);}, {3,1,-1,9,6,1,-1,17}); }
MATRIX_TEST(Sub) { Binary([](auto o,auto l,auto r){Sub(o,l,r);}, {-1,-5,7,-1,-6,-13,15,-1}); }
MATRIX_TEST(PerElementMul) { Binary([](auto o,auto l,auto r){PerElementMul(o,l,r);}, {2,-6,-12,20,0,-42,-56,72}); }

void Broadcast(const std::function<void(Mat*,Mat*,Mat*)>& fn, bool rows, const Values& expected)
{
    auto l = Make(MatrixShape::Matrix(3,2), {1,2,3,4,5,6});
    auto r = rows ? Make(MatrixShape::Matrix(1,2), {10,20}) : Make(MatrixShape::Matrix(3), {10,20,30});
    auto out = Make(MatrixShape::Matrix(3,2));
    fn(out.get(), l.get(), r.get()); Expect(out.get(), expected);
    fn(l.get(), l.get(), r.get()); Expect(l.get(), expected);
}
MATRIX_TEST(BroadcastAdd) { Broadcast([](auto o,auto l,auto r){BroadcastAdd(o,l,r);}, false, {11,22,33,14,25,36}); }
MATRIX_TEST(BroadcastMul) { Broadcast([](auto o,auto l,auto r){BroadcastMul(o,l,r);}, false, {10,40,90,40,100,180}); }
MATRIX_TEST(BroadcastAddRows) { Broadcast([](auto o,auto l,auto r){BroadcastAddRows(o,l,r);}, true, {11,12,13,24,25,26}); }
MATRIX_TEST(BroadcastSubRows) { Broadcast([](auto o,auto l,auto r){BroadcastSubRows(o,l,r);}, true, {-9,-8,-7,-16,-15,-14}); }
MATRIX_TEST(BroadcastMulRows) { Broadcast([](auto o,auto l,auto r){BroadcastMulRows(o,l,r);}, true, {10,20,30,80,100,120}); }
MATRIX_TEST(BroadcastAddThreaded)
{
    ThreadPool pool(2);
    Broadcast([&](auto o,auto l,auto r){BroadcastAdd(o,l,r,&pool,4);}, false, {11,22,33,14,25,36});
}
MATRIX_TEST(PerElementMulThreaded)
{
    ThreadPool pool(2);
    Binary([&](auto o,auto l,auto r){PerElementMul(o,l,r,&pool,3);}, {2,-6,-12,20,0,-42,-56,72});
}

void Scaling(bool scalar, bool add)
{
    auto in = Make(MatrixShape::Image(1u,2,2,2), {1,-2,3,4,5,-6,7,8}), out = Make(MatrixShape::Image(1u,2,2,2));
    auto scale = Make(MatrixShape::Matrix(1), {-2}), bias = Make(MatrixShape::Image(1u,2,2,2), Values(8,3));
    const Values expected = add ? Values{1,7,-3,-5,-7,15,-11,-13} : Values{-2,4,-6,-8,-10,12,-14,-16};
    auto run = [&](Mat* dst) {
        if (add) {
            if (scalar) ScaleAdd(dst,in.get(),-2.0f,bias.get());
            else ScaleAdd(dst,in.get(),scale.get(),bias.get());
        } else {
            if (scalar) Scale(dst,in.get(),-2.0f);
            else Scale(dst,in.get(),scale.get());
        }
    };
    run(out.get()); Expect(out.get(), expected);
    run(in.get()); Expect(in.get(), expected);
}
MATRIX_TEST(ScaleScalar) { Scaling(true,false); }
MATRIX_TEST(ScaleMatrix) { Scaling(false,false); }
MATRIX_TEST(ScaleAddScalar) { Scaling(true,true); }
MATRIX_TEST(ScaleAddMatrix) { Scaling(false,true); }

MATRIX_TEST(MeanSquaredErrorLoss)
{
    auto a = Make(MatrixShape::Image(1u,2,2,2), {1,2,3,4,5,6,7,8}), b = Make(MatrixShape::Image(1u,2,2,2), {2,0,3,6,4,6,10,8});
    Near(MeanSquaredErrorLoss(a.get(),b.get()), 19.0f/8);
}
MATRIX_TEST(TriangleMatrix)
{
    auto out = Make(MatrixShape::Matrix(4,3));
    TriangleMatrix(out.get(),TriangleDirection::Lower,2.0f,-1.0f);
    Expect(out.get(), {2,-1,-1,-1,2,2,-1,-1,2,2,2,-1});
    TriangleMatrix(out.get(),TriangleDirection::Upper,2.0f,-1.0f);
    Expect(out.get(), {2,2,2,2,-1,2,2,2,-1,-1,2,2});
}
MATRIX_TEST(SumColumns)
{
    auto in = Make(MatrixShape::Matrix(3,2), {1,2,3,4,5,6}), out = Make(MatrixShape::Matrix(3));
    SumColumns(out.get(),in.get()); Expect(out.get(), {5,7,9});
}
MATRIX_TEST(SumRows)
{
    auto in = Make(MatrixShape::Matrix(3,2), {1,2,3,4,5,6}), out = Make(MatrixShape::Matrix(1,2));
    SumRows(out.get(),in.get()); Expect(out.get(), {6,15});
}
MATRIX_TEST(SumSpatialDimension)
{
    auto in = Make(MatrixShape::Image(1u,2,2,2), {1,2,3,4,5,6,7,8}), out = Make(MatrixShape::Image(1u,2,2,2), Values(8,-1));
    SumSpatialDimension(out.get(),in.get(),1,{0u,1u,1u,0u});
    Expect(out.get(), {-1,-1,-1,-1,-1,26,-1,-1});
}
MATRIX_TEST(Maximum)
{
    auto in = Make(MatrixShape::Image(1u,2,2,2), {-5,-4,-3,-8,-2,-7,-1,-6}); Near(Maximum(in.get()),-1);
}
MATRIX_TEST(Minimum)
{
    auto in = Make(MatrixShape::Image(1u,2,2,2), {5,4,3,8,2,7,1,6}); Near(Minimum(in.get()),1);
}

MATRIX_TEST(TransposeMat)
{
    auto in = Make(MatrixShape::Image(1u,2,3,2), {1,2,3,4,5,6,7,8,9,10,11,12}), out = Make(MatrixShape::Image(1u,2,2,3));
    TransposeMat(out.get(),in.get()); Expect(out.get(), {1,4,2,5,3,6,7,10,8,11,9,12});
}
MATRIX_TEST(ConcatZ)
{
    auto a = Make(MatrixShape::Matrix(2,2), {1,2,3,4}), b = Make(MatrixShape::Image(1u,2,2,2), {5,6,7,8,9,10,11,12}), out = Make(MatrixShape::Image(1u,3,2,2));
    ConcatZ(out.get(),a.get(),b.get()); Expect(out.get(), {1,2,3,4,5,6,7,8,9,10,11,12});
}
MATRIX_TEST(VConcat)
{
    auto a = Make(MatrixShape::Matrix(2,2), {1,2,3,4}), b = Make(MatrixShape::Matrix(2), {5,6}), out = Make(MatrixShape::Matrix(2,3));
    VConcat(out.get(),a.get(),b.get()); Expect(out.get(), {1,2,3,4,5,6});
}
MATRIX_TEST(VConcatBaseVector)
{
    auto a = Make(MatrixShape::Matrix(2), {1,2}), b = Make(MatrixShape::Matrix(2,2), {3,4,5,6}), out = Make(MatrixShape::Matrix(2,3));
    std::vector<Base*> inputs{a.get(),b.get()};
    VConcat(out.get(),inputs); Expect(out.get(), {1,2,3,4,5,6});
}
MATRIX_TEST(VConcatTypedVector)
{
    auto a = Make(MatrixShape::Matrix(2), {1,2}), b = Make(MatrixShape::Matrix(2,2), {3,4,5,6}), out = Make(MatrixShape::Matrix(2,3));
    const std::vector<Mat*> inputs{a.get(),b.get()};
    VConcat(out.get(),inputs); Expect(out.get(), {1,2,3,4,5,6});
}
MATRIX_TEST(HConcat)
{
    auto a = Make(MatrixShape::Matrix(2,2), {1,2,3,4}), b = Make(MatrixShape::Matrix(1,2), {5,6}), out = Make(MatrixShape::Matrix(3,2));
    HConcat(out.get(),a.get(),b.get()); Expect(out.get(), {1,2,5,3,4,6});
}
MATRIX_TEST(GatherRows)
{
    auto src = Make(MatrixShape::Matrix(2,3), {1,2,3,4,5,6}), idx = Make(MatrixShape::Image(1u,2,1,2), {2,0,1,2}), out = Make(MatrixShape::Image(1u,2,2,2));
    GatherRows(out.get(),src.get(),idx.get()); Expect(out.get(), {5,6,1,2,3,4,5,6});
}
MATRIX_TEST(GatherRowsBatchedSlices)
{
    for (bool shared : {false,true}) {
        const unsigned sourceBatches = shared ? 1u : 2u;
        Values values(6*(sourceBatches+1));
        for (size_t i=0;i<values.size();++i) values[i]=float(i+1);
        auto source=Make(MatrixShape::Image(1u,sourceBatches+1,2,3),values);
        auto indices=Make(MatrixShape::Image(1u,3,1,3),{0,0,0,2,0,2,1,2,0});
        auto output=Make(MatrixShape::Image(1u,4,2,3),Values(24,-7));
        Mat srcView, idxView, outView;
        source->GetPlaneView(&srcView,1); srcView.SetShape(MatrixShape::Image(1u,sourceBatches,2,3));
        indices->GetPlaneView(&idxView,1); idxView.SetShape(MatrixShape::Image(1u,2,1,3));
        output->GetPlaneView(&outView,1); outView.SetShape(MatrixShape::Image(1u,2,2,3));
        GatherRows(&outView,&srcView,&idxView);
#if defined(MATRIX_TEST_Metal)
        for (auto* mat : {&srcView,&idxView,&outView})
            if (mat->m_gInstance->GetMappedPointer(mat->m_dataHandles.m_bufferHandle))
                throw std::runtime_error("GatherRows unexpectedly mapped GPU storage");
#endif
        Values expected(24,-7);
        const unsigned ids[]{2,0,2,1,2,0};
        for (unsigned z=0;z<2;++z) for (unsigned row=0;row<3;++row)
            for (unsigned c=0;c<2;++c)
                expected[6+z*6+row*2+c]=values[6+(shared?0:z)*6+ids[z*3+row]*2+c];
        Expect(output.get(),expected);
        Expect(source.get(),values);
        Expect(indices.get(),{0,0,0,2,0,2,1,2,0});
    }
}
MATRIX_TEST(GatherRowsQueuedCompute)
{
    auto source=Make(MatrixShape::Matrix(2,3),{1,2,3,4,5,6}), indices=Make(MatrixShape::Matrix(1,3),{2,0,2});
    auto output=Make(MatrixShape::Matrix(2,3)), consumed=Make(MatrixShape::Matrix(2,3));
    Scale(source.get(),source.get(),2.0f);
    GatherRows(output.get(),source.get(),indices.get());
    Scale(consumed.get(),output.get(),3.0f);
    Expect(consumed.get(),{30,36,6,12,30,36});
    Expect(source.get(),{2,4,6,8,10,12});
}
MATRIX_TEST(ScatterRows)
{
    auto src = Make(MatrixShape::Image(1u,2,2,2), {1,2,3,4,5,6,7,8}), idx = Make(MatrixShape::Image(1u,2,1,2), {2,0,1,2});
    auto out = Make(MatrixShape::Image(1u,2,2,3), Values(12,-1));
    ScatterRows(out.get(),src.get(),idx.get()); Expect(out.get(), {3,4,-1,-1,1,2,-1,-1,5,6,7,8});
}
MATRIX_TEST(ScatterAddRows)
{
    auto src = Make(MatrixShape::Image(1u,2,2,2), {1,2,3,4,5,6,7,8}), idx = Make(MatrixShape::Image(1u,2,1,2), {1,1,1,2});
    auto out = Make(MatrixShape::Matrix(2,3), Values(6,10));
    ScatterAddRows(out.get(),src.get(),idx.get()); Expect(out.get(), {10,10,19,22,17,18});
}
MATRIX_TEST(ScatterAddRowsBatched)
{
    auto source=Make(MatrixShape::Image(1u,2,2,2),{1,2,3,4,5,6,7,8});
    auto indices=Make(MatrixShape::Image(1u,2,1,2),{1,1,1,2});
    auto output=Make(MatrixShape::Image(1u,2,2,3),Values(12,10.0f));
    ScatterAddRows(output.get(),source.get(),indices.get());
    Expect(output.get(),{10,10,14,16,10,10,10,10,15,16,17,18});
}
MATRIX_TEST(ScatterAddRowsSlices)
{
    auto source=Make(MatrixShape::Image(1u,3,2,2),{9,9,9,9,1,2,3,4,5,6,7,8});
    auto indices=Make(MatrixShape::Image(1u,3,1,2),{0,0,1,1,1,2});
    Values initial(18,-4.0f);std::fill(initial.begin()+6,initial.begin()+12,10.0f);
    auto output=Make(MatrixShape::Image(1u,3,2,3),initial);
    Mat srcView,idxView,outView;
    source->GetPlaneView(&srcView,1);srcView.SetShape(MatrixShape::Image(1u,2,2,2));
    indices->GetPlaneView(&idxView,1);idxView.SetShape(MatrixShape::Image(1u,2,1,2));
    output->GetPlaneView(&outView,1);
    ScatterAddRows(&outView,&srcView,&idxView);
    Expect(output.get(),{-4,-4,-4,-4,-4,-4,10,10,19,22,17,18,-4,-4,-4,-4,-4,-4});
}
MATRIX_TEST(SplitQKV)
{
    auto packed = Make(MatrixShape::Image(1u,2,6,2), {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    auto q = Make(MatrixShape::Image(1u,2,2,2)), k = Make(MatrixShape::Image(1u,2,2,2)), v = Make(MatrixShape::Image(1u,2,2,2));
    SplitQKV(q.get(),k.get(),v.get(),packed.get());
    Expect(q.get(), {1,2,7,8,13,14,19,20}); Expect(k.get(), {3,4,9,10,15,16,21,22});
    Expect(v.get(), {5,6,11,12,17,18,23,24});
}
MATRIX_TEST(MergeQKV)
{
    auto q = Make(MatrixShape::Image(1u,2,2,2), {1,2,3,4,5,6,7,8}), k = Make(MatrixShape::Image(1u,2,2,2), {11,12,13,14,15,16,17,18});
    auto v = Make(MatrixShape::Image(1u,2,2,2), {21,22,23,24,25,26,27,28}), out = Make(MatrixShape::Matrix(12,2));
    MergeQKV(out.get(),q.get(),k.get(),v.get());
    Expect(out.get(), {1,2,5,6,11,12,15,16,21,22,25,26,3,4,7,8,13,14,17,18,23,24,27,28});
}

MATRIX_TEST(MapFunctionZero)
{
    auto out = Make(MatrixShape::Image(1u,2,3,2)); MapFunction_Zero(out.get(),[]{return 4.0f;}); Expect(out.get(),Values(12,4));
}
MATRIX_TEST(MapFunction)
{
    auto in = Make(MatrixShape::Matrix(3,2), {-2,-1,0,1,2,3}), out = Make(MatrixShape::Matrix(3,2));
    MapFunction(out.get(),in.get(),[](float x){return x*x+1;}); Expect(out.get(),{5,2,1,2,5,10});
}
MATRIX_TEST(MapFunctionThreaded)
{
    ThreadPool pool(2);
    auto in = Make(MatrixShape::Matrix(3,2), {-2,-1,0,1,2,3}), out = Make(MatrixShape::Matrix(3,2));
    MapFunction(out.get(),in.get(),[](float x){return x*x+1;},&pool,4);
    Expect(out.get(),{5,2,1,2,5,10});
}
MATRIX_TEST(PerElementFunc)
{
    auto a = Make(MatrixShape::Matrix(3), {1,2,3}), b = Make(MatrixShape::Matrix(3), {4,5,6}), out = Make(MatrixShape::Matrix(3));
    PerElement_Func(out.get(),a.get(),b.get(),[](float x,float y){return x-y*y;});
    Expect(out.get(),{-15,-23,-33});
}

void Unary(const std::function<void(Mat*,Mat*)>& fn, const std::function<float(float)>& reference)
{
    const Values values{-3,-1,-0.1f,0,0.1f,1,2,4}; Values expected;
    for (float x : values) expected.push_back(reference(x));
    auto in = Make(MatrixShape::Image(1u,2,2,2),values), out = Make(MatrixShape::Image(1u,2,2,2));
    fn(out.get(),in.get()); Expect(out.get(),expected);
    fn(in.get(),in.get()); Expect(in.get(),expected);
}
MATRIX_TEST(ReluMat) { Unary([](auto o,auto i){ReluMat(o,i);},[](float x){return std::max(0.0f,x);}); }
MATRIX_TEST(ReluDerivativeMat) { Unary([](auto o,auto i){ReluDerivativeMat(o,i);},[](float x){return x>0?1.0f:0.0f;}); }
MATRIX_TEST(GeluMat) { Unary([](auto o,auto i){GeluMat(o,i);},[](float x){return 0.5f*x*(1+std::erf(x/std::sqrt(2.0f)));}); }
MATRIX_TEST(GeluDerivtiveMat)
{
    Unary([](auto o,auto i){GeluDerivtiveMat(o,i);},[](float x){
        return 0.5f*(1+std::erf(x/std::sqrt(2.0f))) + x*std::exp(-0.5f*x*x)/std::sqrt(2*3.14159265358979323846f);
    });
}

MATRIX_TEST(GeluBackwards)
{
    const Values samples{-12,-8,-3,-1,-0.1f,-0.0001f,0,0.0001f,0.1f,1,2,4,8,12};
    for(uint32_t width : {5u,257u}) {
        const uint32_t plane=width*3,count=plane*2;
        Values input(count),grad(count),expected(count);
        for(uint32_t i=0;i<count;++i) {
            input[i]=samples[i%samples.size()];grad[i]=float(int(i%11)-5)*0.3f;
            const double x=input[i];
            const double derivative=0.5*(1+std::erf(x/std::sqrt(2.0)))+x*std::exp(-0.5*x*x)/std::sqrt(2*3.14159265358979323846);
            expected[i]=float(grad[i]*derivative);
            // An independent central difference of GELU itself.
            const auto gelu=[](double v){return 0.5*v*(1+std::erf(v/std::sqrt(2.0)));};
            Near(expected[i],float(grad[i]*(gelu(x+1e-4)-gelu(x-1e-4))/2e-4));
        }
        Values xb(plane*4,-6),gb(plane*5,-7),ob(plane*4,-8);
        std::copy(input.begin(),input.end(),xb.begin()+plane);
        std::copy(grad.begin(),grad.end(),gb.begin()+2*plane);
        auto x=Make(MatrixShape::Image(1u,4,width,3),xb),g=Make(MatrixShape::Image(1u,5,width,3),gb),out=Make(MatrixShape::Image(1u,4,width,3),ob);
        Mat xv,gv,ov;x->GetPlaneView(&xv,1);g->GetPlaneView(&gv,2);out->GetPlaneView(&ov,1);
        xv.SetShape(MatrixShape::Image(1u,2,width,3));gv.SetShape(MatrixShape::Image(1u,2,width,3));ov.SetShape(MatrixShape::Image(1u,2,width,3));
        auto derivative=Make(MatrixShape::Image(1u,2,width,3)),reference=Make(MatrixShape::Image(1u,2,width,3));
        GeluDerivtiveMat(derivative.get(),&xv);PerElementMul(reference.get(),&gv,derivative.get());
        const auto original=Read(reference.get());
        GeluBackwards(&ov,&xv,&gv);
#if defined(MATRIX_TEST_Metal)
        if(out->m_gInstance->GetMappedPointer(out->m_dataHandles.m_bufferHandle)
           ||x->m_gInstance->GetMappedPointer(x->m_dataHandles.m_bufferHandle)
           ||g->m_gInstance->GetMappedPointer(g->m_dataHandles.m_bufferHandle))
            throw std::runtime_error("GeluBackwards mapped GPU storage");
#endif
        Expect(&ov,original);Expect(&ov,expected);
        std::copy(expected.begin(),expected.end(),ob.begin()+plane);
        Expect(out.get(),ob);Expect(x.get(),xb);Expect(g.get(),gb);
    }
}
MATRIX_TEST(GeluBackwardsQueuedCompute)
{
    auto input=Make(MatrixShape::Image(1u,2,4,2),{-4,-3,-2,-1,0,1,2,3,-0.5f,0.5f,-0.1f,0.1f,-2,2,-5,5});
    auto grad=Make(MatrixShape::Image(1u,2,4,2),{1,-2,0,3,2,-1,0.5f,-0.5f,4,-3,2,-1,0,1,-2,3});
    auto result=Make(MatrixShape::Image(1u,2,4,2)),consumed=Make(MatrixShape::Image(1u,2,4,2));
    auto x=Read(input.get()),g=Read(grad.get());Values expected(x.size());
    for(size_t i=0;i<x.size();++i) {
        const double v=x[i]*0.5;
        expected[i]=float(3*g[i]*(0.5*(1+std::erf(v/std::sqrt(2.0)))+v*std::exp(-0.5*v*v)/std::sqrt(2*3.14159265358979323846)));
    }
    Scale(input.get(),input.get(),0.5f);
    GeluBackwards(result.get(),input.get(),grad.get());
    Scale(consumed.get(),result.get(),3.0f);Expect(consumed.get(),expected);
}

Values SoftmaxReference(const Values& values, unsigned cols)
{
    Values out(values.size());
    for (size_t row = 0; row < values.size(); row += cols) {
        double max = *std::max_element(values.begin()+row, values.begin()+row+cols), sum = 0;
        for (unsigned c=0;c<cols;++c) sum += std::exp(values[row+c]-max);
        for (unsigned c=0;c<cols;++c) out[row+c] = std::exp(values[row+c]-max)/sum;
    }
    return out;
}

Values DropoutMaskReference(MatrixShape dims, float probability, uint32_t seed)
{
    Values values(size_t(dims.width)*dims.height*dims.channels);
    MatrixCPU<float> view(values.data(),dims);
    DropoutMask(&view,probability,seed);
    return values;
}

MATRIX_TEST(DropoutMask)
{
    auto golden = Make(MatrixShape::Matrix(16));
    DropoutMask(golden.get(),0.25f,42u);
    Expect(golden.get(),{1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1});
    const MatrixShape dims = MatrixShape::Image(1u,2,257,3);
    auto mask = Make(dims);
    for (float probability : {-0.1f,0.0f,0.2f,0.75f,1.0f,1.1f}) {
        DropoutMask(mask.get(),probability,12345u);
        Expect(mask.get(),DropoutMaskReference(dims,probability,12345u));
    }
    DropoutMask(mask.get(),0.5f,17u); const auto first = Read(mask.get());
    DropoutMask(mask.get(),0.5f,17u); Expect(mask.get(),first);
    DropoutMask(mask.get(),0.5f,18u);
    if (Read(mask.get())==first) throw std::runtime_error("Different dropout seeds produced identical masks");
    DropoutMask(mask.get(),0.5f); const auto fresh = Read(mask.get());
    DropoutMask(mask.get(),0.5f);
    if (Read(mask.get())==fresh) throw std::runtime_error("Dropout reused its default seed");
}

MATRIX_TEST(DropoutMaskDistribution)
{
    auto mask = Make(MatrixShape::Image(1u,2,65537,1));
    for (float probability : {0.05f,0.2f,0.8f}) {
        DropoutMask(mask.get(),probability,0x12345678u);
        const auto values = Read(mask.get());
        for (float value : values)
            if (value!=0.0f && value!=1.0f) throw std::runtime_error("Dropout mask is not binary");
        const double dropped = std::count(values.begin(),values.end(),0.0f)/double(values.size());
        if (std::abs(dropped-probability)>0.01)
            throw std::runtime_error("Dropout probability is outside tolerance");
        if (std::equal(values.begin(),values.begin()+65537,values.begin()+65537))
            throw std::runtime_error("Dropout repeated its mask across batch planes");
    }
}

MATRIX_TEST(DropoutMaskQueuedSlices)
{
    const MatrixShape dims = MatrixShape::Image(1u,2,17,3);
    constexpr unsigned plane = 17*3;
    auto backing = Make(MatrixShape::Image(1u,4,17,3),Values(plane*4,-77));
    auto input = Make(dims,Values(plane*2,2)), consumed = Make(dims);
    Mat view; backing->GetPlaneView(&view,1); view.SetShape(dims);
#if defined(MATRIX_TEST_Metal)
    auto* manager = backing->m_gInstance;
    manager->WaitForGPU();
    Obj commandBuffer = manager->GetCommandBuffer();
#endif
    DropoutMask(&view,0.2f,71u);
    PerElementMul(consumed.get(),input.get(),&view);
    // Reuse the same constant buffer with different probability and seed.
    DropoutMask(&view,0.8f,99u);
#if defined(MATRIX_TEST_Metal)
    if (manager->GetCommandBuffer()!=commandBuffer || manager->m_lastSubmitted
        || manager->GetMappedPointer(backing->m_dataHandles.m_bufferHandle))
        throw std::runtime_error("Dropout mask submitted or mapped GPU work");
#endif
    auto expected = DropoutMaskReference(dims,0.2f,71u);
    for (auto& value : expected) value *= 2;
    Expect(consumed.get(),expected);
    Values storage(plane*4,-77);
    const auto finalMask = DropoutMaskReference(dims,0.8f,99u);
    std::copy(finalMask.begin(),finalMask.end(),storage.begin()+plane);
    Expect(backing.get(),storage);
    Expect(input.get(),Values(plane*2,2));
}

MATRIX_TEST(ClipGradientNorm)
{
    GradientClipWorkspace<float,Mat> workspace;
    for (unsigned count : {1u,255u,256u,1023u,1024u,1025u,300001u,17u}) {
        for (float magnitude : {0.0f,0.001f,3.0f,1e20f}) {
            Values av(count+22,-77), bv(263+26,-88);
            double squaredNorm=0;
            for(unsigned i=0;i<count;++i) {
                av[11+i]=magnitude*float(int(i%17)-8);
                squaredNorm+=double(av[11+i])*av[11+i];
            }
            for(unsigned i=0;i<263;++i) {
                bv[13+i]=magnitude*float(int(i%13)-6);
                squaredNorm+=double(bv[13+i])*bv[13+i];
            }
            auto a=Make(MatrixShape::Matrix(unsigned(av.size())),av), b=Make(MatrixShape::Matrix(unsigned(bv.size())),bv);
            Mat va,vb,empty;
            va.SetData(a->GetData(),11); va.SetShape(MatrixShape::Matrix(count,1));
            vb.SetData(b->GetData(),13); vb.SetShape(MatrixShape::Matrix(263,1));
            empty.SetShape(MatrixShape::Matrix(0,1));
            const std::vector<Mat*> gradients{&va,&empty,&vb};
            ClipGradientNorm(gradients,1.0f,workspace);
#if defined(MATRIX_TEST_Metal)
            if(a->m_gInstance->GetMappedPointer(a->m_dataHandles.m_bufferHandle)
                ||b->m_gInstance->GetMappedPointer(b->m_dataHandles.m_bufferHandle))
                throw std::runtime_error("Norm clipping mapped gradient storage");
#endif
            const float scale=float(std::min(1.0,1.0/(std::sqrt(squaredNorm)+1e-6)));
            for(unsigned i=0;i<count;++i) av[11+i]*=scale;
            for(unsigned i=0;i<263;++i) bv[13+i]*=scale;
            Expect(a.get(),av); Expect(b.get(),bv);
        }
    }
}

MATRIX_TEST(ClipGradientNormQueuedCompute)
{
    auto a=Make(MatrixShape::Image(1u,2,257,3)),b=Make(MatrixShape::Image(1u,2,17,3)),consumed=Make(MatrixShape::Image(1u,2,257,3));
    GradientClipWorkspace<float,Mat> workspace;
    const std::vector<Mat*> gradients{a.get(),b.get()};
#if defined(MATRIX_TEST_Metal)
    auto* manager=a->m_gInstance; manager->WaitForGPU();
    Obj commandBuffer=manager->GetCommandBuffer();
#endif
    Fill(a.get(),3.0f); Fill(b.get(),-4.0f);
    ClipGradientNorm(gradients,2.0f,workspace);
    Scale(consumed.get(),a.get(),2.0f);
    auto* partials=workspace.partials.get();
    auto* factor=workspace.scale.get();
    // Reuse scratch and its parameter buffers while earlier work is queued.
    Fill(a.get(),0.0f); Fill(b.get(),1.0f);
    ClipGradientNorm(gradients,1000.0f,workspace);
#if defined(MATRIX_TEST_Metal)
    if(manager->GetCommandBuffer()!=commandBuffer || manager->m_lastSubmitted
        ||manager->GetMappedPointer(a->m_dataHandles.m_bufferHandle)
        ||manager->GetMappedPointer(b->m_dataHandles.m_bufferHandle)
        ||manager->GetMappedPointer(workspace.partials->m_dataHandles.m_bufferHandle)
        ||manager->GetMappedPointer(workspace.scale->m_dataHandles.m_bufferHandle))
        throw std::runtime_error("Norm clipping submitted or mapped GPU work");
#endif
    if(workspace.partials.get()!=partials || workspace.scale.get()!=factor)
        throw std::runtime_error("Norm clipping did not reuse scratch storage");
    const double norm=std::sqrt(9.0*a->GetElementCount()+16.0*b->GetElementCount());
    Expect(consumed.get(),Values(a->GetElementCount(),float(6.0*2.0/(norm+1e-6))));
    Expect(a.get(),Values(a->GetElementCount(),0)); Expect(b.get(),Values(b->GetElementCount(),1));
    // Disabled clipping and empty input do not allocate GPU scratch.
    GradientClipWorkspace<float,Mat> disabled;
    ClipGradientNorm(gradients,0.0f,disabled);
    ClipGradientNorm(std::vector<Mat*>{},1.0f,disabled);
    if(disabled.partials || disabled.scale) throw std::runtime_error("Inactive clipping allocated scratch");
}
MATRIX_TEST(Softmax)
{
    const Values values{1000,1001,999,-1000,-1000,-1000};
    auto in = Make(MatrixShape::Matrix(3,2),values), out = Make(MatrixShape::Matrix(3,2));
    const auto expected = SoftmaxReference(values,3);
    Softmax(out.get(),in.get()); Expect(out.get(),expected);
    Softmax(in.get(),in.get()); Expect(in.get(),expected);
}
MATRIX_TEST(AttentionScoresBackwards)
{
    for(uint32_t width : {1u,3u,17u,257u}) for(bool causalProbs : {false,true}) {
        constexpr uint32_t planes=6u;
        const uint32_t plane=width*width;
        Values probabilities(plane*planes), gradients(plane*planes);
        for(uint32_t z=0;z<planes;++z) for(uint32_t row=0;row<width;++row) {
            double sum=0;
            for(uint32_t col=0;col<width;++col) {
                const uint32_t i=z*plane+row*width+col;
                probabilities[i]=causalProbs && col>row ? 0.0f : 0.1f+float((col*7+row*3+z)%13);
                gradients[i]=float(int((i*11+z)%23)-11)*0.17f;
                sum+=probabilities[i];
            }
            for(uint32_t col=0;col<width;++col) probabilities[z*plane+row*width+col]/=float(sum);
        }
        // Different nonzero offsets on each input and output, plus a scalar slice.
        Values pBacking(plane*(planes+3),-9),gBacking(plane*(planes+4),-8),oBacking(plane*(planes+2),-7);
        std::copy(probabilities.begin(),probabilities.end(),pBacking.begin()+2*plane);
        std::copy(gradients.begin(),gradients.end(),gBacking.begin()+3*plane);
        auto p=Make(MatrixShape::Image(1u,planes+3,width,width),pBacking),g=Make(MatrixShape::Image(1u,planes+4,width,width),gBacking);
        auto out=Make(MatrixShape::Image(1u,planes+2,width,width),oBacking),scale=Make(MatrixShape::Image(1u,3,1,1),{-6,0.125f,-6});
        Mat pv,gv,ov,sv;
        p->GetPlaneView(&pv,2);pv.SetShape(MatrixShape::Image(1u,planes,width,width));
        g->GetPlaneView(&gv,3);gv.SetShape(MatrixShape::Image(1u,planes,width,width));
        out->GetPlaneView(&ov,1);ov.SetShape(MatrixShape::Image(1u,planes,width,width));scale->GetPlaneView(&sv,1);
        // Existing three-operation reference runs per plane (CPU softmax backward is 2D).
        auto reference=Make(MatrixShape::Image(1u,planes,width,width)),mask=Make(MatrixShape::Matrix(width,width));
        TriangleMatrix(mask.get(),TriangleDirection::Lower,1.0f,0.0f);
        for(uint32_t z=0;z<planes;++z) {
            Mat rp,rg,ro;pv.GetPlaneView(&rp,z);gv.GetPlaneView(&rg,z);reference->GetPlaneView(&ro,z);
            SoftmaxBackwards(&ro,&rp,&rg);PerElementMul(&ro,&ro,mask.get());Scale(&ro,&ro,&sv);
        }
        const auto expected=Read(reference.get());
        AttentionScoresBackwards(&ov,&pv,&gv,&sv);
#if defined(MATRIX_TEST_Metal)
        if(out->m_gInstance->GetMappedPointer(out->m_dataHandles.m_bufferHandle)
           ||p->m_gInstance->GetMappedPointer(p->m_dataHandles.m_bufferHandle)
           ||g->m_gInstance->GetMappedPointer(g->m_dataHandles.m_bufferHandle))
            throw std::runtime_error("AttentionScoresBackwards mapped GPU storage");
#endif
        std::copy(expected.begin(),expected.end(),oBacking.begin()+plane);
        Expect(out.get(),oBacking);Expect(p.get(),pBacking);Expect(g.get(),gBacking);
        const auto result=Read(&ov);
        for(uint32_t z=0;z<planes;++z) for(uint32_t row=0;row<width;++row) {
            double dot=0;
            for(uint32_t col=0;col<width;++col) {
                const uint32_t i=z*plane+row*width+col;
                dot+=double(probabilities[i])*gradients[i];
            }
            for(uint32_t col=0;col<width;++col) {
                const uint32_t i=z*plane+row*width+col;
                if(col>row && result[i]!=0.0f) throw std::runtime_error("Causal upper triangle must be exactly zero");
                Near(result[i],col>row ? 0.0f : float(probabilities[i]*(gradients[i]-dot)*0.125));
            }
        }
    }
}
MATRIX_TEST(AttentionScoresBackwardsQueuedCompute)
{
    auto p=Make(MatrixShape::Image(1u,2,3,3),{1,0,0,0.3f,0.7f,0,0.2f,0.3f,0.5f,1,0,0,0.4f,0.6f,0,0.1f,0.2f,0.7f});
    auto g=Make(MatrixShape::Image(1u,2,3,3),{1,-2,3,4,-5,6,7,-8,9,-1,2,-3,-4,5,-6,-7,8,-9});
    auto scale=Make(MatrixShape::Matrix(1),{0.5f}),out=Make(MatrixShape::Image(1u,2,3,3)),consumed=Make(MatrixShape::Image(1u,2,3,3));
    const auto probs=Read(p.get()),grad=Read(g.get());Values expected(18,0);
    for(unsigned row=0;row<6;++row) {
        double dot=0;for(unsigned c=0;c<3;++c) dot+=double(probs[row*3+c])*grad[row*3+c]*2;
        for(unsigned c=0;c<=row%3;++c) expected[row*3+c]=float(3*0.5*probs[row*3+c]*(grad[row*3+c]*2-dot));
    }
    Scale(g.get(),g.get(),2.0f);
    AttentionScoresBackwards(out.get(),p.get(),g.get(),scale.get());
    Scale(consumed.get(),out.get(),3.0f);Expect(consumed.get(),expected);
}

// Use borrowed CPU matrices as the reference so all backends share the same
// loss/gradient contract, including the different index and one-hot scaling.
void CrossEntropyParity(bool oneHot, bool allIgnored = false)
{
    CategoricalCrossEntropyWithLogits<float, Mat> loss;
    CategoricalCrossEntropyWithLogits<float, MatrixCPU<float>> reference;
    for (uint32_t width : {1u, 3u, 65u, 257u})
    {
        if (oneHot && width == 1u) continue; // Width one denotes index targets in the API.
        for (bool broadcast : {false, true})
        for (uint32_t steps : {1u, 137u})
        {
            constexpr uint32_t batches = 3u;
            const uint32_t plane = width * steps;
            const uint32_t targetWidth = oneHot ? width : 1u;
            const uint32_t targetBatches = broadcast ? 1u : batches;
            const uint32_t targetPlane = targetWidth * steps;
            Values logits(plane * batches), targets(targetPlane * targetBatches, 0);
            for (uint32_t row = 0; row < steps * batches; ++row)
                for (uint32_t x = 0; x < width; ++x)
                    logits[row * width + x] = (row % 2 ? 1000.0f : -1000.0f)
                        + float(int((x * 7 + row * 11) % 29) - 14) * 2.0f;
            for (uint32_t row = 0; row < steps * targetBatches; ++row)
            {
                const uint32_t label = (row * 5 + 1) % width;
                if (oneHot) targets[row * width + label] = 1.0f;
                else targets[row] = allIgnored || row % 5 == 0 ? -1.0f : float(label);
            }
            Values expected(logits.size());
            MatrixCPU<float> cpuLogits(logits.data(), MatrixShape::Image(1u,batches,width,steps));
            MatrixCPU<float> cpuTargets(targets.data(), MatrixShape::Image(1u,targetBatches,targetWidth,steps));
            MatrixCPU<float> cpuGradient(expected.data(), MatrixShape::Image(1u,batches,width,steps));
            reference.Gradient(&cpuGradient, &cpuTargets, &cpuLogits);
            const float expectedLoss = reference.Loss(&cpuTargets, &cpuLogits);

            // Different offsets for logits, targets and output; padding must survive.
            Values lBacking(plane * (batches + 3), -7), tBacking(targetPlane * (targetBatches + 4), -8);
            Values oBacking(plane * (batches + 2), -9);
            std::copy(logits.begin(), logits.end(), lBacking.begin() + 2 * plane);
            std::copy(targets.begin(), targets.end(), tBacking.begin() + 3 * targetPlane);
            auto l = Make(MatrixShape::Image(1u,batches + 3,width,steps), lBacking);
            auto t = Make(MatrixShape::Image(1u,targetBatches + 4,targetWidth,steps), tBacking);
            auto o = Make(MatrixShape::Image(1u,batches + 2,width,steps), oBacking);
            Mat lv, tv, ov;
            l->GetPlaneView(&lv, 2); lv.SetShape(MatrixShape::Image(1u,batches,width,steps));
            t->GetPlaneView(&tv, 3); tv.SetShape(MatrixShape::Image(1u,targetBatches,targetWidth,steps));
            o->GetPlaneView(&ov, 1); ov.SetShape(MatrixShape::Image(1u,batches,width,steps));
            loss.Gradient(&ov, &tv, &lv);
#if defined(MATRIX_TEST_Metal)
            for (auto* mat : {l.get(), t.get(), o.get()})
                if (mat->m_gInstance->GetMappedPointer(mat->m_dataHandles.m_bufferHandle))
                    throw std::runtime_error("Cross-entropy gradient mapped GPU storage");
#endif
            Near(loss.Loss(&tv, &lv), expectedLoss);
#if defined(MATRIX_TEST_Metal)
            if (l->m_gInstance->GetMappedPointer(l->m_dataHandles.m_bufferHandle)
                || t->m_gInstance->GetMappedPointer(t->m_dataHandles.m_bufferHandle))
                throw std::runtime_error("Cross-entropy loss downloaded inputs");
#endif
            std::copy(expected.begin(), expected.end(), oBacking.begin() + plane);
            Expect(o.get(), oBacking);
            Expect(l.get(), lBacking); Expect(t.get(), tBacking);
        }
    }
}
MATRIX_TEST(CrossEntropyIndexTargets) { CrossEntropyParity(false); }
MATRIX_TEST(CrossEntropyOneHotTargets) { CrossEntropyParity(true); }
MATRIX_TEST(CrossEntropyIgnoredTargets) { CrossEntropyParity(false, true); }
MATRIX_TEST(CrossEntropyLossAccumulation)
{
    for (bool oneHot : {false, true})
    {
        const Values initial{1,-2,3,4,-5,6,7,-8,9};
        Values labels = oneHot ? Values{1,0,0,0,1,0,0,0,1} : Values{0,-1,2};
        const uint32_t targetWidth = oneHot ? 3u : 1u;
        auto logits = Make(MatrixShape::Image(1u,2,3,3), Values(18, -7));
        auto targets = Make(MatrixShape::Image(1u,2,targetWidth,3), Values(labels.size()*2, -8));
        Mat lv, tv;
        logits->GetPlaneView(&lv,1); targets->GetPlaneView(&tv,1);
        Write(&lv, initial); Write(&tv, labels);
        CategoricalCrossEntropyWithLogits<float, Mat> loss;
        CategoricalCrossEntropyWithLogits<float, MatrixCPU<float>> reference;
        Values values = initial;
        MatrixCPU<float> cpuLogits(values.data(), MatrixShape::Matrix(3,3));
        MatrixCPU<float> cpuTargets(labels.data(), MatrixShape::Matrix(targetWidth,3));
        const float first = reference.Loss(&cpuTargets,&cpuLogits);
        for (auto& value : values) value *= 0.5f;
        const float second = reference.Loss(&cpuTargets,&cpuLogits);
#if defined(MATRIX_TEST_Metal)
        auto* gpu = logits->m_gInstance;
        gpu->WaitForGPU();
        Obj cb = gpu->GetCommandBuffer();
#endif
        loss.ResetAccumulatedLoss();
        loss.AccumulateLoss(&tv,&lv);
        Scale(&lv,&lv,0.5f);
        loss.AccumulateLoss(&tv,&lv);
#if defined(MATRIX_TEST_Metal)
        if (gpu->GetCommandBuffer() != cb || gpu->m_lastSubmitted)
            throw std::runtime_error("Accumulating loss submitted GPU work");
        if (gpu->GetMappedPointer(logits->m_dataHandles.m_bufferHandle)
            || gpu->GetMappedPointer(targets->m_dataHandles.m_bufferHandle))
            throw std::runtime_error("Accumulating loss downloaded inputs");
#endif
        Near(float(loss.GetAccumulatedLossMean()), (first+second)/2);
        Near(float(loss.GetAccumulatedLossMean()), (first+second)/2);
        loss.ResetAccumulatedLoss();
        Near(float(loss.GetAccumulatedLossMean()),0.0f);
        loss.AccumulateLoss(&tv,&lv);
        Near(float(loss.GetAccumulatedLossMean()),second);
        loss.ResetAccumulatedLoss();
        Fill(&tv,oneHot ? 0.0f : -1.0f);
        loss.AccumulateLoss(&tv,&lv);
        Near(float(loss.GetAccumulatedLossMean()),0.0f);
        Values expectedLogits(18,-7), expectedTargets(labels.size()*2,-8);
        std::copy(values.begin(),values.end(),expectedLogits.begin()+9);
        std::fill(expectedTargets.begin()+labels.size(),expectedTargets.end(),oneHot ? 0.0f : -1.0f);
        Expect(logits.get(),expectedLogits); Expect(targets.get(),expectedTargets);
    }
}
MATRIX_TEST(CrossEntropyQueuedCompute)
{
    constexpr uint32_t width = 3, steps = 3, batches = 2;
    auto logits = Make(MatrixShape::Image(1u,batches,width,steps), {1,-2,3,4,-5,6,7,-8,9,-1,2,-3,-4,5,-6,-7,8,-9});
    auto targets = Make(MatrixShape::Matrix(1,steps), {0,0,0});
    auto first = Make(MatrixShape::Image(1u,batches,width,steps)), second = Make(MatrixShape::Image(1u,batches,width,steps));
    auto consumed = Make(MatrixShape::Image(1u,batches,width,steps));
    CategoricalCrossEntropyWithLogits<float, Mat> loss;
    Values values = Read(logits.get());
    for (float& value : values) value *= 0.5f;
    Values labels(steps, 1), expected(values.size());
    MatrixCPU<float> cpuLogits(values.data(), MatrixShape::Image(1u,batches,width,steps));
    MatrixCPU<float> cpuTargets(labels.data(), MatrixShape::Matrix(1,steps));
    MatrixCPU<float> cpuGradient(expected.data(), MatrixShape::Image(1u,batches,width,steps));
    CategoricalCrossEntropyWithLogits<float, MatrixCPU<float>> reference;
    reference.Gradient(&cpuGradient, &cpuTargets, &cpuLogits);
    for (float& value : expected) value *= 2.0f;
    Scale(logits.get(), logits.get(), 0.5f);
    Fill(targets.get(), 1.0f); // Targets are produced on the GPU, not downloaded for counting.
#if defined(MATRIX_TEST_Metal)
    auto* manager = logits->m_gInstance;
    Obj recorded = manager->m_commandBuffer;
#endif
    loss.Gradient(first.get(), targets.get(), logits.get());
    Scale(consumed.get(), first.get(), 2.0f);
    Fill(targets.get(), -1.0f);
    loss.Gradient(second.get(), targets.get(), logits.get()); // Reuses the GPU count buffer.
#if defined(MATRIX_TEST_Metal)
    if (manager->m_commandBuffer != recorded || manager->m_lastSubmitted)
        throw std::runtime_error("Cross-entropy gradient unexpectedly submitted or waited");
    for (auto* mat : {logits.get(), targets.get(), first.get(), second.get()})
        if (manager->GetMappedPointer(mat->m_dataHandles.m_bufferHandle))
            throw std::runtime_error("Queued cross-entropy mapped GPU storage");
#endif
    Expect(consumed.get(), expected);
    Expect(second.get(), Values(values.size(), 0));
    Near(loss.Loss(targets.get(), logits.get()), 0.0f);
}

MATRIX_TEST(SoftmaxBackwards)
{
    const Values values{1,2,3,-2,1,0}, grad{1,-2,3,4,0,-1};
    auto soft = Make(MatrixShape::Matrix(3,2),SoftmaxReference(values,3)), dy = Make(MatrixShape::Matrix(3,2),grad), out = Make(MatrixShape::Matrix(3,2));
    Values expected(6);
    for (size_t i=0;i<6;++i) {
        auto plus=values, minus=values; plus[i]+=0.002f; minus[i]-=0.002f;
        auto p=SoftmaxReference(plus,3), m=SoftmaxReference(minus,3);
        for (size_t j=0;j<6;++j) expected[i] += grad[j]*(p[j]-m[j])/0.004f;
    }
    SoftmaxBackwards(out.get(),soft.get(),dy.get()); Expect(out.get(),expected,1e-3f);
}

// Double precision reference and finite differences check forward and gradient math
// independently of the library's CPU implementation.
Values Normalize(const Values& input, const Values& gamma, const Values& beta, double eps=1e-5)
{
    Values out(input.size()); const size_t cols=gamma.size();
    for (size_t row=0;row<input.size();row+=cols) {
        double mean=0,var=0;
        for(size_t c=0;c<cols;++c) mean+=input[row+c]; mean/=cols;
        for(size_t c=0;c<cols;++c) var+=std::pow(input[row+c]-mean,2); var/=cols;
        for(size_t c=0;c<cols;++c) out[row+c]=(input[row+c]-mean)/std::sqrt(var+eps)*gamma[c]+beta[c];
    }
    return out;
}
MATRIX_TEST(BackwardReductionWidths)
{
    // Cover idle lanes, full groups and multiple elements per lane across multiple rows.
    for (unsigned width : {1u, 3u, 255u, 256u, 257u, 384u, 513u}) {
        Values values(width * 4), grad(values.size()), gammaValues(width);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = i / width == 0 ? 0.5f : float(int(i % 17) - 8) * 0.125f;
            grad[i] = float(int(i % 13) - 6) * 0.25f;
        }
        for (unsigned c = 0; c < width; ++c)
            gammaValues[c] = float(int(c % 7) - 3) * 0.2f;
        const Values probabilities = SoftmaxReference(values, width);
        Values expectedSoft(values.size()), expectedNorm(values.size());
        for (size_t row = 0; row < values.size(); row += width) {
            double dot = 0, mean = 0, variance = 0;
            for (unsigned c = 0; c < width; ++c) {
                dot += double(grad[row+c]) * probabilities[row+c];
                mean += values[row+c];
            }
            mean /= width;
            for (unsigned c = 0; c < width; ++c)
                variance += std::pow(values[row+c] - mean, 2);
            const double invStd = 1 / std::sqrt(variance / width + 1e-5);
            double meanGrad = 0, meanGradHat = 0;
            for (unsigned c = 0; c < width; ++c) {
                const double g = double(grad[row+c]) * gammaValues[c];
                meanGrad += g / width;
                meanGradHat += g * (values[row+c] - mean) * invStd / width;
            }
            for (unsigned c = 0; c < width; ++c) {
                expectedSoft[row+c] = probabilities[row+c] * (grad[row+c] - dot);
                const double hat = (values[row+c] - mean) * invStd;
                expectedNorm[row+c] = invStd * (double(grad[row+c]) * gammaValues[c]
                                               - meanGrad - hat * meanGradHat);
            }
        }
        auto in = Make(MatrixShape::Matrix(width,4), values), dy = Make(MatrixShape::Matrix(width,4), grad);
        auto soft = Make(MatrixShape::Matrix(width,4), probabilities), out = Make(MatrixShape::Matrix(width,4));
        SoftmaxBackwards(out.get(), soft.get(), dy.get());
        Expect(out.get(), expectedSoft);
        auto gamma = Make(MatrixShape::Matrix(width), gammaValues), dg = Make(MatrixShape::Matrix(width));
        auto hat = Make(MatrixShape::Matrix(width,4), Normalize(values, Values(width,1), Values(width,0)));
        LayerNormBackwardsOp(out.get(), dy.get(), in.get(), gamma.get(), dg.get(),
                             static_cast<Mat*>(nullptr), hat.get());
        Expect(out.get(), expectedNorm);
    }
}
MATRIX_TEST(LayerNormOp)
{
    const Values values{1,2,4,-3,1,5,2,2,2,7,0,-2}, g{1,2,-1}, b{0.5f,-1,2};
    auto in=Make(MatrixShape::Image(1u,2,3,2),values), gamma=Make(MatrixShape::Matrix(3),g), beta=Make(MatrixShape::Matrix(3),b);
    auto out=Make(MatrixShape::Image(1u,2,3,2)), hat=Make(MatrixShape::Image(1u,2,3,2));
    for(bool bias : {false,true}) for(bool cache : {false,true}) {
        LayerNormOp(out.get(),in.get(),gamma.get(),bias?beta.get():nullptr,cache?hat.get():nullptr);
        Expect(out.get(),Normalize(values,g,bias?b:Values(3,0)));
        if(cache) Expect(hat.get(),Normalize(values,Values(3,1),Values(3,0)));
    }
}
MATRIX_TEST(LayerNormBackwardsOp)
{
    const Values values{1,2,4,-3,1,5,2,2.5f,3,7,0,-2}, g{1,2,-1}, grad{1,-2,3,2,0,-1,1,2,3,-1,3,2};
    auto in=Make(MatrixShape::Image(1u,2,3,2),values), gamma=Make(MatrixShape::Matrix(3),g), dy=Make(MatrixShape::Image(1u,2,3,2),grad);
    auto hat=Make(MatrixShape::Image(1u,2,3,2),Normalize(values,Values(3,1),Values(3,0)));
    auto dx=Make(MatrixShape::Image(1u,2,3,2)), dg=Make(MatrixShape::Matrix(3)), db=Make(MatrixShape::Matrix(3));
    const auto normalized=Read(hat.get()); Values expectedG(3,0),expectedB(3,0),expectedX(12,0);
    for(size_t i=0;i<12;++i) { expectedG[i%3]+=grad[i]*normalized[i]; expectedB[i%3]+=grad[i]; }
    for(size_t i=0;i<12;++i) {
        auto plus=values,minus=values; plus[i]+=0.001f; minus[i]-=0.001f;
        auto p=Normalize(plus,g,Values(3,0)),m=Normalize(minus,g,Values(3,0));
        for(size_t j=0;j<12;++j) expectedX[i]+=grad[j]*(p[j]-m[j])/0.002f;
    }
    for(bool bias : {false,true}) {
        LayerNormBackwardsOp(dx.get(),dy.get(),in.get(),gamma.get(),dg.get(),bias?db.get():nullptr,hat.get());
        Expect(dx.get(),expectedX,0.003f); Expect(dg.get(),expectedG); if(bias) Expect(db.get(),expectedB);
    }
}

MATRIX_TEST(LayerNormCachedStatistics)
{
    for (uint32_t width : {1u, 3u, 257u, 384u}) for (bool bias : {false,true}) {
        Values values(width*4), grad(width*4), gammaValues(width);
        for (size_t i=0;i<values.size();++i) {
            values[i] = i < width ? 2.0f : float(int(i%19)-9)*0.13f;
            grad[i] = float(int(i%11)-5)*0.17f;
        }
        for (size_t i=0;i<width;++i) gammaValues[i]=0.5f+float(i%7)*0.1f;
        auto in=Make(MatrixShape::Image(1u,2,width,2),values), dy=Make(MatrixShape::Image(1u,2,width,2),grad);
        auto gamma=Make(MatrixShape::Matrix(width),gammaValues), beta=Make(MatrixShape::Matrix(width),Values(width,0.3f));
        auto out=Make(MatrixShape::Image(1u,2,width,2)), hat=Make(MatrixShape::Image(1u,2,width,2));
        auto stats=Make(MatrixShape::Image(1u,4,1,2),Values(8,-7.0f));
        Mat statsView; stats->GetPlaneView(&statsView,1); statsView.SetShape(MatrixShape::Image(1u,2,1,2));
        LayerNormOp(out.get(),in.get(),gamma.get(),bias?beta.get():nullptr,hat.get(),1e-5f,&statsView);
        Expect(out.get(),Normalize(values,gammaValues,Values(width,bias?0.3f:0.0f)));
        Values expectedStats(8,-7.0f);
        for (uint32_t row=0;row<4;++row) {
            double mean=0,var=0;
            for(uint32_t c=0;c<width;++c) mean+=double(values[row*width+c])/width;
            for(uint32_t c=0;c<width;++c) var+=std::pow(values[row*width+c]-mean,2)/width;
            expectedStats[2+row]=float(1/std::sqrt(var+1e-5));
        }
        Expect(stats.get(),expectedStats);
        auto dx=Make(MatrixShape::Image(1u,2,width,2)), dg=Make(MatrixShape::Matrix(width)), db=Make(MatrixShape::Matrix(width));
        LayerNormBackwardsOp(dx.get(),dy.get(),in.get(),gamma.get(),dg.get(),bias?db.get():nullptr,hat.get());
        const auto expectedX=Read(dx.get()), expectedG=Read(dg.get());
        const auto expectedB=bias?Read(db.get()):Values{};
        // Cached backward must use forward statistics, not reread or reduce input.
        Write(in.get(),Values(values.size(),std::numeric_limits<float>::quiet_NaN()));
        LayerNormBackwardsOp(dx.get(),dy.get(),in.get(),gamma.get(),dg.get(),bias?db.get():nullptr,hat.get(),1e-5f,&statsView);
        Expect(dx.get(),expectedX,0.002f); Expect(dg.get(),expectedG);
        if(bias) Expect(db.get(),expectedB);
        Expect(stats.get(),expectedStats);
    }
}

MATRIX_TEST(ParameterReductions)
{
    for(unsigned width : {1u,17u,33u,384u}) for(unsigned rows : {63u,127u,128u,257u,1024u}) {
        const unsigned plane=width*rows;
        Values dyValues(plane*4),hatValues(plane*4);
        for(size_t i=0;i<dyValues.size();++i) {
            dyValues[i]=float(int(i%19)-9)*0.017f;
            hatValues[i]=float(int(i%13)-6)*0.023f;
        }
        auto dy=Make(MatrixShape::Image(1u,4,width,rows),dyValues),hat=Make(MatrixShape::Image(1u,4,width,rows),hatValues);
        auto input=Make(MatrixShape::Image(1u,2,width,rows),Values(plane*2,0));
        auto gamma=Make(MatrixShape::Matrix(width),Values(width,1)),stats=Make(MatrixShape::Image(1u,2,1,rows),Values(rows*2,1));
        auto dx=Make(MatrixShape::Image(1u,2,width,rows));
        auto dg=Make(MatrixShape::Image(1u,3,width,1),Values(width*3,-77));
        auto db=Make(MatrixShape::Image(1u,3,width,1),Values(width*3,-88));
        auto sum=Make(MatrixShape::Image(1u,3,width,1),Values(width*3,-99));
        Mat yv,hv,gv,bv,sv;
        dy->GetPlaneView(&yv,1);yv.SetShape(MatrixShape::Image(1u,2,width,rows));
        hat->GetPlaneView(&hv,1);hv.SetShape(MatrixShape::Image(1u,2,width,rows));
        dg->GetPlaneView(&gv,1);db->GetPlaneView(&bv,1);sum->GetPlaneView(&sv,1);
        Values expectedG(width*3,-77),expectedB(width*3,-88),expectedSum(width*3,-99);
        for(unsigned col=0;col<width;++col) {
            double gs=0,bs=0,ss=0;
            for(unsigned row=0;row<rows*2;++row) {
                const unsigned index=plane+row*width+col;
                gs+=double(dyValues[index])*hatValues[index];bs+=dyValues[index];
                if(row<rows) ss+=dyValues[index]; // SumColumns preserves its first-plane semantics.
            }
            expectedG[width+col]=float(gs);expectedB[width+col]=float(bs);expectedSum[width+col]=float(ss);
        }
        SumColumns(&sv,&yv);Expect(sum.get(),expectedSum);
        for(bool beta : {false,true}) {
            LayerNormBackwardsOp(dx.get(),&yv,input.get(),gamma.get(),&gv,beta ? &bv : nullptr,&hv,1e-5f,stats.get());
            Expect(dg.get(),expectedG);
            if(beta) Expect(db.get(),expectedB);
            else Expect(db.get(),Values(width*3,-88));
        }
        Expect(dy.get(),dyValues);Expect(hat.get(),hatValues);
    }
}

MATRIX_TEST(ParameterReductionsQueuedCompute)
{
    auto input=Make(MatrixShape::Matrix(33,257)),sum=Make(MatrixShape::Matrix(33)),consumer=Make(MatrixShape::Matrix(33));
#if defined(MATRIX_TEST_Metal)
    auto* manager=input->m_gInstance;manager->WaitForGPU();
    Obj cb=manager->GetCommandBuffer();
#endif
    Fill(input.get(),2.0f);SumColumns(sum.get(),input.get());Scale(consumer.get(),sum.get(),3.0f);
#if defined(MATRIX_TEST_Metal)
    if(manager->GetCommandBuffer()!=cb || manager->m_lastSubmitted
        ||manager->GetMappedPointer(input->m_dataHandles.m_bufferHandle)
        ||manager->GetMappedPointer(sum->m_dataHandles.m_bufferHandle))
        throw std::runtime_error("Parallel reduction mapped or submitted GPU work");
#endif
    Expect(consumer.get(),Values(33,1542.0f));
}

template<TransposeMode Mode>
void MultiplyVariants()
{
    constexpr bool tl=Mode==TransposeMode::Left || Mode==TransposeMode::Both;
    constexpr bool tr=Mode==TransposeMode::Right || Mode==TransposeMode::Both;
    auto a=tl?Make(MatrixShape::Matrix(2,3),{1,4,2,5,3,6}):Make(MatrixShape::Matrix(3,2),{1,2,3,4,5,6});
    auto b=tr?Make(MatrixShape::Matrix(3,4),{1,5,9,2,6,10,3,7,11,4,8,12}):Make(MatrixShape::Matrix(4,3),{1,2,3,4,5,6,7,8,9,10,11,12});
    auto out=Make(MatrixShape::Matrix(4,2)); const Values expected{38,44,50,56,83,98,113,128};
    Mul<Mode>(out.get(),a.get(),b.get()); Expect(out.get(),expected);
    MatMul_Strided<Mode>(out.get(),a.get(),b.get(),a->GetWidth(),1,b->GetWidth(),1,4,1);
    Expect(out.get(),expected);
}
MATRIX_TEST(Mul) { MultiplyVariants<TransposeMode::None>(); }
MATRIX_TEST(MulTransposeLeft) { MultiplyVariants<TransposeMode::Left>(); }
MATRIX_TEST(MulTransposeRight) { MultiplyVariants<TransposeMode::Right>(); }
MATRIX_TEST(MulTransposeBoth) { MultiplyVariants<TransposeMode::Both>(); }

template<TransposeMode Mode>
void TiledMultiplyEdges()
{
    constexpr bool tl = Mode == TransposeMode::Left || Mode == TransposeMode::Both;
    constexpr bool tr = Mode == TransposeMode::Right || Mode == TransposeMode::Both;
    // Partial tiles, exact tiles, and a long reduction across multiple tiles.
    for (const auto dims : {MatrixShape::Image(1u,37,33,19), MatrixShape::Image(1u,64,48,32), MatrixShape::Image(1u,129,17,65), MatrixShape::Image(1u,137,129,133)}) {
        const unsigned m = dims.height, n = dims.width, k = dims.channels;
        const unsigned ax = tl ? m : k, ay = tl ? k : m;
        const unsigned bx = tr ? k : n, by = tr ? n : k;
        for (bool strided : {false, true}) {
            const unsigned col = strided ? 2 : 1;
            const unsigned ar = ax * col + (strided ? 3 : 0);
            const unsigned br = bx * col + (strided ? 5 : 0);
            const unsigned cr = n * col + (strided ? 7 : 0);
            Values av(ar*ay*3), bv(br*by*3), cv(cr*m*3, -77), expected = cv;
            for (size_t i = 0; i < av.size(); ++i) av[i] = float(int(i%23)-11)*0.0625f;
            for (size_t i = 0; i < bv.size(); ++i) bv[i] = float(int(i%19)-9)*0.125f;
            const unsigned batches = strided ? 2 : 1;
            for (unsigned z = 1; z <= batches; ++z)
                for (unsigned y = 0; y < m; ++y) for (unsigned x = 0; x < n; ++x) {
                    double sum = 0;
                    for (unsigned inner = 0; inner < k; ++inner)
                        sum += double(av[z*ar*ay + (tl?inner:y)*ar + (tl?y:inner)*col])
                             * bv[z*br*by + (tr?x:inner)*br + (tr?inner:x)*col];
                    expected[z*cr*m + y*cr + x*col] = sum;
                }
            auto a = Make(MatrixShape::Image(1u,3,ar,ay), av), b = Make(MatrixShape::Image(1u,3,br,by), bv), out = Make(MatrixShape::Image(1u,3,cr,m), cv);
            Mat aView, bView, outView;
            a->GetPlaneView(&aView,1); aView.SetShape(MatrixShape::Image(1u,batches,ax,ay));
            b->GetPlaneView(&bView,1); bView.SetShape(MatrixShape::Image(1u,batches,bx,by));
            out->GetPlaneView(&outView,1); outView.SetShape(MatrixShape::Image(1u,batches,n,m));
            if (strided)
                MatMul_Strided<Mode>(&outView,&aView,&bView,ar,col,br,col,cr,col);
            else
                Mul<Mode>(&outView,&aView,&bView);
            // The complete backing buffer also checks that partial tiles preserve padding.
            Expect(out.get(), expected);
            Expect(a.get(), av); Expect(b.get(), bv);
        }
    }
}
MATRIX_TEST(MulTiledEdges) { TiledMultiplyEdges<TransposeMode::None>(); }
MATRIX_TEST(MulTiledEdgesTransposeLeft) { TiledMultiplyEdges<TransposeMode::Left>(); }
MATRIX_TEST(MulTiledEdgesTransposeRight) { TiledMultiplyEdges<TransposeMode::Right>(); }
MATRIX_TEST(MulTiledEdgesTransposeBoth) { TiledMultiplyEdges<TransposeMode::Both>(); }

template<TransposeMode Mode>
void ExplicitBatchStrides()
{
    constexpr bool tl = Mode == TransposeMode::Left || Mode == TransposeMode::Both;
    constexpr bool tr = Mode == TransposeMode::Right || Mode == TransposeMode::Both;
    constexpr unsigned batches = 3;
    for (bool tiled : {false, true}) for (unsigned layout = 0; layout < 4; ++layout) {
        const unsigned m = tiled ? 65 : 3, n = tiled ? 97 : 5, k = tiled ? 193 : 7;
        const unsigned ax = tl ? m : k, ay = tl ? k : m;
        const unsigned bx = tr ? k : n, by = tr ? n : k;
        auto strides = [&](unsigned x, unsigned y) -> MatrixStrides {
            if (layout == 1) return {x * batches * 2 + 3, 2, x * 2}; // interleaved heads
            if (layout == 2) return {1, y + 2, (y + 2) * x + 7}; // column-major
            return {x * 2 + 3, 2, (x * 2 + 3) * y + 7}; // padded planes
        };
        auto as = strides(ax, ay), bs = strides(bx, by), cs = strides(n, m);
        if (layout == 3) as.channel = 0; // broadcast reads
        auto size = [&](unsigned x, unsigned y, MatrixStrides st, unsigned offset) {
            return offset + (batches-1)*st.channel + (y-1)*st.row + (x-1)*st.column + 14;
        };
        constexpr unsigned ao = 11, bo = 17, co = 23;
        Values av(size(ax,ay,as,ao)), bv(size(bx,by,bs,bo));
        Values cv(size(n,m,cs,co), -77), expected = cv;
        for (size_t i=0;i<av.size();++i) av[i] = float(int(i%23)-11)*0.0625f;
        for (size_t i=0;i<bv.size();++i) bv[i] = float(int(i%19)-9)*0.125f;
        for (unsigned z=0;z<batches;++z) for (unsigned y=0;y<m;++y) for (unsigned x=0;x<n;++x) {
            double sum = 0;
            for (unsigned inner=0;inner<k;++inner)
                sum += double(av[ao+z*as.channel+(tl?inner:y)*as.row+(tl?y:inner)*as.column])
                     * bv[bo+z*bs.channel+(tr?x:inner)*bs.row+(tr?inner:x)*bs.column];
            expected[co+z*cs.channel+y*cs.row+x*cs.column] = sum;
        }
        auto a=Make(MatrixShape::Matrix(unsigned(av.size())),av), b=Make(MatrixShape::Matrix(unsigned(bv.size())),bv);
        auto out=Make(MatrixShape::Matrix(unsigned(cv.size())),cv);
        Mat aView, bView, outView;
        aView.SetData(a->GetData(),ao); aView.SetShape(MatrixShape::Image(1u,batches,ax,ay));
        bView.SetData(b->GetData(),bo); bView.SetShape(MatrixShape::Image(1u,batches,bx,by));
        outView.SetData(out->GetData(),co); outView.SetShape(MatrixShape::Image(1u,batches,n,m));
        MatMul_Batched<Mode>(&outView,&aView,&bView,cs,as,bs);
#if defined(MATRIX_TEST_Metal)
        if (outView.m_gInstance->GetMappedPointer(outView.m_dataHandles.m_bufferHandle))
            throw std::runtime_error("Batched multiplication fell back to CPU mapping");
#endif
        Expect(out.get(),expected); // includes gaps, neighboring heads and both sentinels
        Expect(a.get(),av); Expect(b.get(),bv); // reads must preserve all source storage
    }
}
MATRIX_TEST(MatMulBatchedExplicitStrides) { ExplicitBatchStrides<TransposeMode::None>(); }
MATRIX_TEST(MatMulBatchedExplicitStridesLeft) { ExplicitBatchStrides<TransposeMode::Left>(); }
MATRIX_TEST(MatMulBatchedExplicitStridesRight) { ExplicitBatchStrides<TransposeMode::Right>(); }
MATRIX_TEST(MatMulBatchedExplicitStridesBoth) { ExplicitBatchStrides<TransposeMode::Both>(); }

MATRIX_TEST(MulQueuedSlices)
{
    constexpr unsigned n = 128;
    auto a = Make(MatrixShape::Image(1u,2,n,n)), b = Make(MatrixShape::Image(1u,2,n,n)), out = Make(MatrixShape::Image(1u,2,n,n));
    Mat av, bv, ov;
    for (unsigned z = 0; z < 2; ++z) {
        a->GetPlaneView(&av,z); b->GetPlaneView(&bv,z); out->GetPlaneView(&ov,z);
        Fill(&av,float(1+2*z)); Fill(&bv,float(2+2*z));
        Mul(&ov,&av,&bv);
    }
    // No readback between GPU producers, slice reuse, and the consumer.
    Scale(out.get(),out.get(),3.0f);
    Values expected(n*n*2,768.0f);
    std::fill(expected.begin()+n*n,expected.end(),4608.0f);
    Expect(out.get(),expected);
}

MATRIX_TEST(CopySliceOffsets)
{
    auto source = Make(MatrixShape::Image(1u,3,3,2), {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18});
    auto output = Make(MatrixShape::Image(1u,4,3,2), Values(24,-1));
    Mat srcView, outView;
    source->GetPlaneView(&srcView,1); output->GetPlaneView(&outView,2);
    Copy(&outView,&srcView);
    Expect(output.get(), {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,7,8,9,10,11,12,-1,-1,-1,-1,-1,-1});
    Expect(source.get(), {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18});
    Mat sameBufferSource, sameBufferDestination;
    output->GetPlaneView(&sameBufferSource,2); output->GetPlaneView(&sameBufferDestination,0);
    Copy(&sameBufferDestination,&sameBufferSource);
    Expect(output.get(), {7,8,9,10,11,12,-1,-1,-1,-1,-1,-1,7,8,9,10,11,12,-1,-1,-1,-1,-1,-1});
    // Exercise the mapped CPU-copy branch as well as the GPU copy above.
    source->DataRead(); output->DataRead();
    source->GetPlaneView(&srcView,2); output->GetPlaneView(&outView,3);
    Copy(&outView,&srcView);
    Expect(output.get(), {7,8,9,10,11,12,-1,-1,-1,-1,-1,-1,7,8,9,10,11,12,13,14,15,16,17,18});
}

MATRIX_TEST(RemoveBorrowedView)
{
    auto parent = Make(MatrixShape::Image(1u,3,3,2), {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18});
    auto& manager = MatrixManager<float,Mat>::Instance();
    auto view = manager.GetSkeleton("Borrowed slice");
    parent->GetPlaneView(view.get(),1);
    auto retained = view;
    manager.RemoveMatrix(view.get());
    view.Reset();
    Expect(retained.get(), {7,8,9,10,11,12});
    retained.Reset();
    Expect(parent.get(), {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18});
}

template<TransposeMode Mode>
void StridedBatches()
{
    // Exercise column-major physical storage and two row-major batches.
    constexpr bool tl=Mode==TransposeMode::Left || Mode==TransposeMode::Both;
    constexpr bool tr=Mode==TransposeMode::Right || Mode==TransposeMode::Both;
    const unsigned ax=tl?2:3, ay=tl?3:2, bx=tr?3:4, by=tr?4:3;
    Values av(12),bv(24),expected(16);
    for(unsigned z=0;z<2;++z) {
        for(unsigned y=0;y<ay;++y) for(unsigned x=0;x<ax;++x) av[z*6+y+x*ay]=1+x+2*y+z;
        for(unsigned y=0;y<by;++y) for(unsigned x=0;x<bx;++x) bv[z*12+y+x*by]=1+2*x+y-z;
        for(unsigned i=0;i<2;++i) for(unsigned j=0;j<4;++j) {
            float sum=0;
            for(unsigned k=0;k<3;++k) sum+=av[z*6+(tl?k:i)+(tl?i:k)*ay]*bv[z*12+(tr?j:k)+(tr?k:j)*by];
            expected[z*8+i+j*2]=sum;
        }
    }
    auto a=Make(MatrixShape::Image(1u,2,ax,ay),av),b=Make(MatrixShape::Image(1u,2,bx,by),bv),out=Make(MatrixShape::Image(1u,2,4,2));
    // Batched plane stride in the API is rowStride*rows, so use
    // row-major matrices for the batched call, then column-major for a single plane.
    Values rowA(12),rowB(24),rowExpected(16);
    for(unsigned z=0;z<2;++z) {
        for(unsigned y=0;y<ay;++y) for(unsigned x=0;x<ax;++x) rowA[z*6+y*ax+x]=av[z*6+y+x*ay];
        for(unsigned y=0;y<by;++y) for(unsigned x=0;x<bx;++x) rowB[z*12+y*bx+x]=bv[z*12+y+x*by];
        for(unsigned y=0;y<2;++y) for(unsigned x=0;x<4;++x) rowExpected[z*8+y*4+x]=expected[z*8+y+x*2];
    }
    Write(a.get(),rowA); Write(b.get(),rowB);
    MatMul_Strided<Mode>(out.get(),a.get(),b.get(),ax,1,bx,1,4,1); Expect(out.get(),rowExpected);
    a->SetShape(MatrixShape::Matrix(ax,ay)); b->SetShape(MatrixShape::Matrix(bx,by)); out->SetShape(MatrixShape::Matrix(4,2));
    Write(a.get(),Values(av.begin(),av.begin()+6)); Write(b.get(),Values(bv.begin(),bv.begin()+12));
    MatMul_Strided<Mode>(out.get(),a.get(),b.get(),1,ay,1,by,1,2);
    Expect(out.get(),Values(expected.begin(),expected.begin()+8));
}
MATRIX_TEST(MatMulStrided) { StridedBatches<TransposeMode::None>(); }
MATRIX_TEST(MatMulStridedTransposeLeft) { StridedBatches<TransposeMode::Left>(); }
MATRIX_TEST(MatMulStridedTransposeRight) { StridedBatches<TransposeMode::Right>(); }
MATRIX_TEST(MatMulStridedTransposeBoth) { StridedBatches<TransposeMode::Both>(); }

// Lower-level shared multiplication entry points are tested on each matrix backend.
void SharedMul(int variant)
{
    auto a=Make(MatrixShape::Matrix(3,2),{1,2,3,4,5,6}),b=Make(MatrixShape::Matrix(4,3),{1,2,3,4,5,6,7,8,9,10,11,12}),out=Make(MatrixShape::Matrix(4,2));
    ThreadPool pool(2); Base *o=out.get(),*l=a.get(),*r=b.get();
    switch(variant) {
        case 0: MulNoDynamicAlloc(o,l,r); break;
        case 1: MulBlock(o,l,r,2); break;
        case 2: MulBlock(o,l,r,2,3); break;
        case 3: MulBlock(o,l,r,2,3,&pool); break;
        case 4: MulStrided(o,l,r,3,1,4,1,4,1); break;
        case 5: MulStrided(l->DataRead(),2,3,3,1,r->DataRead(),3,4,4,1,o->DataWrite(),4,1); break;
        case 6: MulStridedBlock(o,l,r,3,1,4,1,4,1,2,3); break;
        case 7: MulStridedBlock(o,l,r,3,1,4,1,4,1,2,3,&pool); break;
        case 8: MulStridedBlock(l->DataRead(),2,3,3,1,r->DataRead(),3,4,4,1,o->DataWrite(),4,1,2); break;
    }
    Expect(out.get(),{38,44,50,56,83,98,113,128});
}
MATRIX_TEST(MulNoDynamicAlloc) { SharedMul(0); }
MATRIX_TEST(MulBlock) { SharedMul(1); }
MATRIX_TEST(MulBlockThreaded) { SharedMul(2); }
MATRIX_TEST(MulBlockThreadPool) { SharedMul(3); }
MATRIX_TEST(MulStridedMatrix) { SharedMul(4); }
MATRIX_TEST(MulStridedRaw) { SharedMul(5); }
MATRIX_TEST(MulStridedBlockMatrix) { SharedMul(6); }
MATRIX_TEST(MulStridedBlockThreadPool) { SharedMul(7); }
MATRIX_TEST(MulStridedBlockRaw) { SharedMul(8); }
MATRIX_TEST(MulLarge)
{
    Values av(35*33),bv(37*35),expected(37*33,0);
    for(size_t i=0;i<av.size();++i) av[i]=static_cast<int>(i%7)-3;
    for(size_t i=0;i<bv.size();++i) bv[i]=static_cast<int>(i%5)-2;
    for(unsigned y=0;y<33;++y) for(unsigned x=0;x<37;++x)
        for(unsigned k=0;k<35;++k) expected[y*37+x]+=av[y*35+k]*bv[k*37+x];
    auto a=Make(MatrixShape::Matrix(35,33),av),b=Make(MatrixShape::Matrix(37,35),bv),out=Make(MatrixShape::Matrix(37,33)); ThreadPool pool(2);
    Mul(out.get(),a.get(),b.get(),8,2,&pool); Expect(out.get(),expected);
    MatMul_Strided(out.get(),a.get(),b.get(),35,1,37,1,37,1,8,2,&pool); Expect(out.get(),expected);
}

// Scatter-based reference describes both convolution directions and their gradients.
void Convolution(bool transpose, bool backward)
{
    for(unsigned stride : {1u,2u}) for(unsigned dilation : {1u,2u}) for(unsigned padding : {0u,1u}) {
        const unsigned iw=4,ih=3,kw=2,kh=2,channels=2;
        const unsigned ow=transpose?(iw-1)*stride+dilation*(kw-1)+1-2*padding:
            (iw+2*padding-dilation*(kw-1)-1)/stride+1;
        const unsigned oh=transpose?(ih-1)*stride+dilation*(kh-1)+1-2*padding:
            (ih+2*padding-dilation*(kh-1)-1)/stride+1;
        Values iv(iw*ih*channels),kv(kw*kh*channels),gy(ow*oh),forward(ow*oh,0.5f);
        for(size_t i=0;i<iv.size();++i) iv[i]=static_cast<int>(i%7)-2;
        for(size_t i=0;i<kv.size();++i) kv[i]=(static_cast<int>(i%5)-2)*0.5f;
        for(size_t i=0;i<gy.size();++i) gy[i]=(static_cast<int>(i%3)-1)*0.25f;
        Values dw(kv.size(),1),dx(iv.size(),2);
        const unsigned w=transpose?iw:ow,h=transpose?ih:oh;
        for(unsigned c=0;c<channels;++c) for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x)
            for(unsigned ky=0;ky<kh;++ky) for(unsigned kx=0;kx<kw;++kx) {
                const int px=int(x*stride+kx*dilation)-int(padding),py=int(y*stride+ky*dilation)-int(padding);
                if(px<0 || py<0 || px>=int(transpose?ow:iw) || py>=int(transpose?oh:ih)) continue;
                const unsigned ii=c*iw*ih+(transpose?y*iw+x:py*iw+px);
                const unsigned oi=transpose?py*ow+px:y*ow+x,ki=c*kw*kh+ky*kw+kx;
                forward[oi]+=iv[ii]*kv[ki]; dw[ki]+=iv[ii]*gy[oi]; dx[ii]+=kv[ki]*gy[oi];
            }
        auto in=Make(MatrixShape::Image(1u,channels,iw,ih),iv),kernel=Make(MatrixShape::Image(1u,channels,kw,kh),kv),bias=Make(MatrixShape::Matrix(2),{-1,0.5f});
        auto out=Make(MatrixShape::Image(1u,2,ow,oh),Values(ow*oh*2,-99)),dy=Make(MatrixShape::Image(1u,2,ow,oh),Values(ow*oh*2,0));
        Values grad(ow*oh*2,0); std::copy(gy.begin(),gy.end(),grad.begin()+ow*oh); Write(dy.get(),grad);
        if(backward) {
            for(int optional=0;optional<3;++optional) {
                auto wg=Make(MatrixShape::Image(1u,channels,kw,kh),Values(kv.size(),1)),ig=Make(MatrixShape::Image(1u,channels,iw,ih),Values(iv.size(),2));
                Mat* wp=optional==1?nullptr:wg.get(); Mat* ip=optional==2?nullptr:ig.get();
                if(transpose) Conv2DTransposeSingleChannelBackwards(dy.get(),in.get(),kernel.get(),wp,ip,1,stride,dilation,padding);
                else Conv2DSingleChannelBackwards(dy.get(),in.get(),kernel.get(),wp,ip,1,stride,dilation,padding);
                if(wp) Expect(wp,dw); if(ip) Expect(ip,dx);
            }
        } else {
            for(bool withBias : {true,false}) {
                if(transpose) Conv2DTransposeSingleChannel(out.get(),in.get(),kernel.get(),withBias?bias.get():nullptr,1,stride,dilation,padding);
                else Conv2DSingleChannel(out.get(),in.get(),kernel.get(),withBias?bias.get():nullptr,1,stride,dilation,padding);
                Values expected(ow*oh*2,-99);
                for(unsigned i=0;i<ow*oh;++i) expected[ow*oh+i]=forward[i]-(withBias?0:0.5f);
                Expect(out.get(),expected);
            }
        }
    }
}
MATRIX_TEST(Conv2DSingleChannel) { Convolution(false,false); }
MATRIX_TEST(Conv2DSingleChannelBackwards) { Convolution(false,true); }
MATRIX_TEST(Conv2DTransposeSingleChannel) { Convolution(true,false); }
MATRIX_TEST(Conv2DTransposeSingleChannelBackwards) { Convolution(true,true); }

void Adam(bool decay)
{
    Values pv{1,-2,3,0},gv{0.5f,-1,0,2},mv{0,0.2f,-0.1f,0},vv{0,0.3f,0.2f,0};
    auto p=Make(MatrixShape::Matrix(2,2),pv),g=Make(MatrixShape::Matrix(2,2),gv),m=Make(MatrixShape::Matrix(2,2),mv),v=Make(MatrixShape::Matrix(2,2),vv);
    const float lr=0.01f,b1=0.9f,b2=0.99f,eps=1e-6f,wd=decay?0.1f:0;
    for(int step=1;step<=3;++step) {
        for(size_t i=0;i<4;++i) {
            mv[i]=b1*mv[i]+(1-b1)*gv[i]; vv[i]=b2*vv[i]+(1-b2)*gv[i]*gv[i];
            pv[i]+=lr*(mv[i]/(1-std::pow(b1,step)))/(std::sqrt(vv[i]/(1-std::pow(b2,step)))+eps)-lr*wd*pv[i];
        }
        const float b1Power=std::pow(b1,step),b2Power=std::pow(b2,step);
        if(decay) AdamWUpdate(p.get(),g.get(),m.get(),v.get(),lr,b1,b2,b1Power,b2Power,eps,wd);
        else AdamUpdate(p.get(),g.get(),m.get(),v.get(),lr,b1,b2,b1Power,b2Power,eps);
        Expect(p.get(),pv); Expect(m.get(),mv); Expect(v.get(),vv);
    }
}
MATRIX_TEST(AdamUpdate) { Adam(false); }
MATRIX_TEST(AdamWUpdate) { Adam(true); }
MATRIX_TEST(ReparameterizeMat)
{
    Values means(2048),vars(2048);
    for(size_t i=0;i<means.size();++i) {means[i]=float(i%3)-1; vars[i]=i%4==0?-20:i%4==1?10:0;}
    auto mu=Make(MatrixShape::Matrix(2048),means),lv=Make(MatrixShape::Matrix(2048),vars),z=Make(MatrixShape::Matrix(2048)),eps=Make(MatrixShape::Matrix(2048));
    ReparameterizeMat(z.get(),eps.get(),mu.get(),lv.get()); const auto noise=Read(eps.get());
    Values expected(2048); double mean=0,variance=0;
    for(size_t i=0;i<noise.size();++i) {
        if(!std::isfinite(noise[i])) throw std::runtime_error("Nonfinite noise");
        expected[i]=means[i]+std::exp(0.5f*std::clamp(vars[i],-10.0f,4.0f))*noise[i];
        mean+=noise[i]; variance+=noise[i]*noise[i];
    }
    Expect(z.get(),expected); mean/=noise.size(); variance=variance/noise.size()-mean*mean;
    // Broad statistical bounds catch a missing/constant RNG without flaky sampling thresholds.
    if(std::abs(mean)>0.15 || variance<0.7 || variance>1.3) throw std::runtime_error("Noise is not approximately standard normal");
}
MATRIX_TEST(ReparameterizeBackwardsMat)
{
    const Values means{1,-2,0,3},vars{-20,10,0,2},noise{0.5f,-1,2,0},grad{2,-1,0,3};
    auto mu=Make(MatrixShape::Matrix(4),means),lv=Make(MatrixShape::Matrix(4),vars),e=Make(MatrixShape::Matrix(4),noise),dz=Make(MatrixShape::Matrix(4),grad),dm=Make(MatrixShape::Matrix(4)),dl=Make(MatrixShape::Matrix(4));
    Values expectedM(4),expectedL(4);
    for(size_t i=0;i<4;++i) {
        const float sigma=std::exp(0.5f*std::clamp(vars[i],-10.0f,4.0f));
        expectedM[i]=grad[i]-0.1f*means[i]; expectedL[i]=grad[i]*noise[i]*0.5f*sigma-0.05f*(sigma*sigma-1);
    }
    ReparameterizeBackwardsMat(dm.get(),dl.get(),dz.get(),mu.get(),lv.get(),e.get(),0.1f);
    Expect(dm.get(),expectedM); Expect(dl.get(),expectedL);
}

MATRIX_TEST(MatrixAccess)
{
    auto mat=Make(MatrixShape::Image(1u,2,3,2),{1,2,3,4,5,6,7,8,9,10,11,12});
    Near(mat->GetValue(2,1),6); Near(mat->GetValue(1,0,1),8);
    mat->SetValue(2,1,-6); mat->SetValue(1,0,1,-8);
    Near(mat->GetValue(2,1),-6); Near(mat->GetValue(1,0,1),-8);
    auto row=mat->GetRow(1),column=mat->GetColumn(1);
    if(row!=Values{4,5,-6} || column!=Values{2,5}) throw std::runtime_error("Row/column access failed");
    std::vector<char> raw; mat->GetRawData(raw);
    if(raw.size()!=12*sizeof(float)) throw std::runtime_error("Raw data size mismatch");
    Values copied(12); std::memcpy(copied.data(),raw.data(),raw.size());
    if(copied!=Values{1,2,3,4,5,-6,7,-8,9,10,11,12}) throw std::runtime_error("Raw data mismatch");
}
MATRIX_TEST(SliceZ)
{
    auto parent=Make(MatrixShape::Image(1u,3,3,2),{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18});
    Mat slice; parent->GetPlaneView(&slice,1);
    Expect(&slice,{7,8,9,10,11,12});
    Fill(&slice,-2.0f); Expect(&slice,Values(6,-2));
    Expect(parent.get(),{1,2,3,4,5,6,-2,-2,-2,-2,-2,-2,13,14,15,16,17,18});
}
MATRIX_TEST(QueuedScalarParameters)
{
    auto mat = Make(MatrixShape::Matrix(4), {1,2,3,4});
    Scale(mat.get(), mat.get(), 2.0f);
    Scale(mat.get(), mat.get(), 3.0f);
    Expect(mat.get(), {6,12,18,24});
}
MATRIX_TEST(QueuedSliceParameters)
{
    auto parent = Make(MatrixShape::Image(1u,2,2,1), {1,2,3,4});
    {
        Mat slice;
        parent->GetPlaneView(&slice, 0);
        Fill(&slice, 7.0f);
    }
    // Reusing the temporary view's constant buffer must not alter the first fill.
    {
        Mat slice;
        parent->GetPlaneView(&slice, 1);
        Fill(&slice, 9.0f);
    }
    Expect(parent.get(), {7,7,9,9});
}
MATRIX_TEST(QueuedStagingRead)
{
    auto mat = Make(MatrixShape::Matrix(4), {1,2,3,4});
    Scale(mat.get(), mat.get(), 2.0f);
#if defined(MATRIX_TEST_Metal)
    auto* manager = mat->m_gInstance;
    manager->CopyGPUToStaging(mat->m_dataHandles.m_bufferHandle);
    const auto* data = static_cast<const float*>(manager->MapBuffer(mat->m_dataHandles.m_bufferHandle));
    for (size_t i = 0; i < 4; ++i) Near(data[i], 2.0f * (i + 1));
    manager->UnmapBuffer(mat->m_dataHandles.m_bufferHandle);
#else
    Expect(mat.get(), {2,4,6,8});
#endif
}
MATRIX_TEST(QueuedStagingWrite)
{
    auto mat = Make(MatrixShape::Matrix(4), {1,2,3,4});
    Scale(mat.get(), mat.get(), 2.0f);
#if defined(MATRIX_TEST_Metal)
    auto* manager = mat->m_gInstance;
    auto* data = static_cast<float*>(manager->MapBuffer(mat->m_dataHandles.m_bufferHandle));
    std::fill(data, data + 4, 7.0f);
    manager->CopyStagingToGPU(mat->m_dataHandles.m_bufferHandle);
    manager->UnmapBuffer(mat->m_dataHandles.m_bufferHandle);
#else
    Write(mat.get(), {7,7,7,7});
#endif
    Expect(mat.get(), {7,7,7,7});
}
MATRIX_TEST(QueuedComputeCopyClear)
{
    constexpr uint32_t count = 4096;
    auto source = Make(MatrixShape::Matrix(count), Values(count, 2.0f));
    auto copied = Make(MatrixShape::Matrix(count), Values(count, -1.0f));
    auto output = Make(MatrixShape::Matrix(count), Values(count, -1.0f));
    // No CPU access between submissions: both compute/blit ordering and the
    // source's reuse after a copy must be preserved.
    for (unsigned i = 0; i < 8; ++i)
    {
        Fill(source.get(), 2.0f);
        Scale(source.get(), source.get(), 3.0f);
        Copy(copied.get(), source.get());
        Clear(source.get());
        Add(output.get(), copied.get(), source.get());
        Copy(copied.get(), output.get());
    }
    Expect(copied.get(), Values(count, 6.0f));
    Expect(source.get(), Values(count, 0.0f));
}
MATRIX_TEST(ReadOnlyMapping)
{
    auto mat = Make(MatrixShape::Matrix(4), {1,2,3,4});
    Near(mat->DataRead()[0], 1.0f);
#if defined(MATRIX_TEST_Metal)
    auto* manager = mat->m_gInstance;
    if (manager->IsMappingDirty(mat->m_dataHandles.m_bufferHandle))
        throw std::runtime_error("Read-only mapping marked dirty");
    // Poison staging through the internal API to detect an unnecessary upload.
    auto* staging = static_cast<float*>(manager->GetMappedPointer(mat->m_dataHandles.m_bufferHandle));
    staging[0] = -99.0f;
#endif
    Commit(mat.get());
    Scale(mat.get(), mat.get(), 2.0f);
    Expect(mat.get(), {2,4,6,8});
}
MATRIX_TEST(MappingWriteUpgrade)
{
    auto parent = Make(MatrixShape::Image(1u,2,2,1), {1,2,3,4});
    parent->DataRead();
    Mat slice; parent->GetPlaneView(&slice, 1);
    slice.DataWrite()[0] = 9.0f;
    // A write through a view must dirty the parent's backing-buffer mapping.
    Commit(parent.get());
    Scale(parent.get(), parent.get(), 2.0f);
    Expect(parent.get(), {2,4,18,8});
}
MATRIX_TEST(MappedCopyDestination)
{
    auto input = Make(MatrixShape::Matrix(4), {1,2,3,4});
    auto output = Make(MatrixShape::Matrix(4), {-1,-1,-1,-1});
    input->DataRead(); output->DataRead();
    Copy(output.get(), input.get());
    Commit(input.get()); Commit(output.get());
    Scale(output.get(), output.get(), 2.0f);
    Expect(output.get(), {2,4,6,8});
}
MATRIX_TEST(GeluActivationDispatch)
{
    auto input = Make(MatrixShape::Matrix(4), {-1,0,1,2}), output = Make(MatrixShape::Matrix(4));
    Gelu<float,Mat> activation;
    activation.activateMat(output.get(), input.get());
#if defined(MATRIX_TEST_Metal)
    if (output->m_gInstance->GetMappedPointer(output->m_dataHandles.m_bufferHandle))
        throw std::runtime_error("GELU forward used CPU mapping");
#endif
    Values expected;
    for (float x : Values{-1,0,1,2}) expected.push_back(0.5f*x*(1+std::erf(x/std::sqrt(2.0f))));
    Expect(output.get(), expected);
    activation.activateMat(output.get(), input.get(), nullptr);
    Expect(output.get(), expected);
}
MATRIX_TEST(CopyQueuedChain)
{
    auto source=Make(MatrixShape::Image(1u,3,3,2),{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18});
    auto middle=Make(MatrixShape::Image(1u,4,3,2),Values(24,-1)),output=Make(MatrixShape::Image(1u,3,3,2),Values(18,-2));
    Mat src,mid,dst;
    source->GetPlaneView(&src,1);middle->GetPlaneView(&mid,2);output->GetPlaneView(&dst,1);
#if defined(MATRIX_TEST_Metal)
    auto* manager=source->m_gInstance;manager->WaitForGPU();
    Obj recorded=manager->GetCommandBuffer();
#endif
    Scale(&src,&src,2.0f);
    Copy(&mid,&src);
    Copy(&dst,&mid);
#if defined(MATRIX_TEST_Metal)
    if(manager->GetCommandBuffer()!=recorded || manager->m_lastSubmitted)
        throw std::runtime_error("GPU copies submitted or replaced the current command buffer");
#endif
    Scale(&dst,&dst,3.0f);
    Expect(output.get(),{-2,-2,-2,-2,-2,-2,42,48,54,60,66,72,-2,-2,-2,-2,-2,-2});
    Expect(middle.get(),{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,14,16,18,20,22,24,-1,-1,-1,-1,-1,-1});
    Expect(source.get(),{1,2,3,4,5,6,14,16,18,20,22,24,13,14,15,16,17,18});
}
MATRIX_TEST(WaitAfterRecordedCopies)
{
    auto source=Make(MatrixShape::Matrix(4096),Values(4096,3.0f));
    auto middle=Make(MatrixShape::Matrix(4096),Values(4096,-1.0f)),output=Make(MatrixShape::Matrix(4096),Values(4096,-2.0f));
#if defined(MATRIX_TEST_Metal)
    auto* manager=source->m_gInstance;manager->WaitForGPU();
    Obj recorded=manager->GetCommandBuffer();Send<void>(recorded,"retain");
#endif
    Copy(middle.get(),source.get());Copy(output.get(),middle.get());
#if defined(MATRIX_TEST_Metal)
    if(manager->GetCommandBuffer()!=recorded || manager->m_lastSubmitted)
        throw std::runtime_error("Copy-only work must stay recorded until submission");
    manager->WaitForGPU();
    const auto status=Send<uint64_t>(recorded,"status");Release(recorded);
    if(status!=4u) throw std::runtime_error("WaitForGPU did not complete recorded copies");
    manager->WaitForGPU();
#endif
    Expect(output.get(),Values(4096,3.0f));Expect(middle.get(),Values(4096,3.0f));
}

MATRIX_TEST(WaitAfterSubmittedCopy)
{
    auto source = Make(MatrixShape::Matrix(4096), Values(4096, 3.0f));
    auto output = Make(MatrixShape::Matrix(4096), Values(4096, -1.0f));
    Scale(source.get(), source.get(), 2.0f);
    Copy(output.get(), source.get());
#if defined(MATRIX_TEST_Metal)
    auto* manager = source->m_gInstance;
    // Explicit submission followed by a wait with no newly recorded work.
    manager->SubmitRecordedWork();
    Obj submitted = manager->m_lastSubmitted;
    Send<void>(submitted, "retain");
    manager->WaitForGPU();
    auto status = Send<uint64_t>(submitted, "status");
    Release(submitted);
    if (status != 4u) // MTLCommandBufferStatusCompleted
        throw std::runtime_error("WaitForGPU did not complete submitted copy");
    manager->WaitForGPU(); // Repeated waits without new work remain valid.
#endif
    Expect(output.get(), Values(4096, 6.0f));
}
} // namespace

int main(int argc,char** argv)
{
    if(argc!=2) {std::cerr<<"Usage: MatrixTests_<backend> <case>\n";return 2;}
    const auto test=std::find_if(Tests().begin(),Tests().end(),[&](const Test& t){return t.first==argv[1];});
    if(test==Tests().end()) {std::cerr<<"Unknown test: "<<argv[1]<<'\n';return 2;}
    try {
        // Managers log during static teardown, so construct the logger first.
        LOG_INFO() << "Starting matrix test";
#if defined(MATRIX_TEST_Metal)
        if(!MetalManager::Instance()->GetQueue()) {std::cout<<"SKIP: No Metal device available\n";return 77;}
#elif defined(MATRIX_TEST_DX11)
        if(!DirectX11Manager::Instance()->GetContext()) {std::cout<<"SKIP: No DirectX 11 device available\n";return 77;}
#endif
        test->second(); std::cout<<"PASS: "<<test->first<<'\n';return 0;
    } catch(const std::exception& error) {
        std::cerr<<"FAIL: "<<test->first<<": "<<error.what()<<'\n';return 1;
    }
}
