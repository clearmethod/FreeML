#include "LayerTestSupport.h"
#include <Loading/SaveLoader.h>
#include <chrono>

using namespace LayerTesting;
using M = TestMatrix;

template<class Exception, class F> void Reject(F action, const char* message)
{
    bool rejected = false;
    try { action(); } catch (const Exception&) { rejected = true; }
    Require(rejected, message);
}

int main()
{
    try {
        LOG_INFO() << "Starting four-dimensional shape/view tests";
#if defined(LAYER_TEST_Metal)
        if (!MetalManager::Instance()->GetQueue()) return 77;
#elif defined(LAYER_TEST_DX11)
        if (!DirectX11Manager::Instance()->GetContext()) return 77;
#endif
        const auto shape = MatrixShape::Image(3,2,5,4);
        Require(shape.ElementCount() == 120 && shape.ByteSize(sizeof(float)) == 480, "Shape size");
        uint64_t index = 0;
        for (unsigned b=0;b<3;++b) for (unsigned c=0;c<2;++c)
            for (unsigned y=0;y<4;++y) for (unsigned x=0;x<5;++x)
                Require(shape.Index(b,c,x,y) == index++, "Row-major addressing");
        Reject<std::out_of_range>([&]{shape.Index(3,0,0,0);}, "Batch bounds");
        Reject<std::out_of_range>([&]{shape.Index(0,2,0,0);}, "Channel bounds");
        Reject<std::invalid_argument>([&]{shape.Concatenated(shape,static_cast<MatrixAxis>(99));},"Invalid concatenation axis");
        Reject<std::overflow_error>([]{MatrixShape::Image(UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX).ElementCount();}, "64-bit overflow");
        Reject<std::overflow_error>([]{MatrixShape::Matrix(65536,65536).CheckedElementCount();}, "32-bit overflow");
        auto& manager = MatrixManager<float,M>::Instance();
        Reject<std::overflow_error>([&]{manager.AllocateMatrix(MatrixShape::Matrix(65536,16384),"TooLarge");}, "Allocation byte overflow");
        auto source = manager.AllocateMatrix(shape,"FourDimensionalStorage");
        Values values(120);for(unsigned i=0;i<120;++i) values[i]=float(i);
        Write(source.get(),values);
        Near(source->GetValue(2,1,4,3),119.0f,"Explicit coordinate read");
        auto batch = manager.BatchView(source,1);
        auto channel = manager.ChannelView(batch,0,1);
        auto reshaped = manager.ReshapeView(channel,MatrixShape::Matrix(4,5));
        Require(batch->GetShape() == MatrixShape::Image(1,2,5,4),"Batch view shape");
        Values expected(values.begin()+60,values.begin()+80);
        Expect(Read(reshaped.get()),expected,"Nested view addressing");
        source.Reset(); batch.Reset(); channel.Reset();
        Expect(Read(reshaped.get()),expected,"View backing allocation lifetime");
        Reject<std::invalid_argument>([&]{manager.ReshapeView(reshaped,MatrixShape::Matrix(3,5));},"Invalid reshape");
        Reject<std::out_of_range>([&]{manager.CreateView(reshaped,MatrixShape::Matrix(21));},"View bounds");
        auto first = manager.CreateView(reshaped,MatrixShape::Matrix(1));
        Near(first->GetValue(0,0),60.0f,"Zero relative offset retains parent offset");
        auto alternate=manager.AllocateMatrix(MatrixShape::Matrix(30),"AlternateBacking");
        Write(alternate.get(),Values(30,7.0f));
        manager.UpdateView(first,alternate,MatrixShape::Matrix(1),13u);
        Near(first->GetValue(0,0),7.0f,"Rebound view offset");
        manager.UpdateView(first,alternate,MatrixShape::Matrix(1),0u);
        Require(first->GetViewParent()==alternate.get(),"Rebound view parent lifetime");
        Near(first->GetValue(0,0),7.0f,"Rebound zero offset");
        auto nested=manager.CreateView(first,MatrixShape::Matrix(1));
        Reject<std::invalid_argument>([&]{manager.UpdateView(first,nested,MatrixShape::Matrix(1));},"View ownership cycle");
        alternate.Reset();
        Near(first->GetValue(0,0),7.0f,"Rebound allocation retained");
        auto small = manager.GetScratch(MatrixShape::Image(1,2,5,4));
        auto large = manager.GetScratch(shape);
        Require(small != large && large->GetElementCount()==120,"Scratch key includes batch axis");

        // SaveLoader writes into a caller-selected directory. Only remove this test's directory.
        const auto directory = std::filesystem::current_path() / ("MatrixShapeTest-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        struct Cleanup { std::filesystem::path path; ~Cleanup(){std::filesystem::remove_all(path);} } cleanup{directory};
        std::string outputName="ShapeOutput";
        std::unique_ptr<Datablob<float,M>> blob(InitActivationBlob<float,M>(shape,outputName,1,false));
        ActivationLayer<float,Identity<float>,M> layer;
        layer.SetName("ShapeRoundTrip");
        Write(blob->GetMatrix("Output_0"),values);
        SaveLoader<float,M> loader;loader.Save(&layer,blob.get(),directory.string());
        blob.reset(); // Force a byte load instead of reusing a live matrix GUID.
        Datablob<float,M> restoredBlob;
        std::unique_ptr<Layer<float,M>> restored(loader.Load(&restoredBlob,"ShapeRoundTrip",directory.string()));
        Require(bool(restored),"Layer checkpoint load");
        auto* restoredOutput=restoredBlob.GetMatrix("Output_0");
        Require(restoredOutput && restoredOutput->GetShape()==shape,"Checkpoint shape round trip");
        Expect(Read(restoredOutput),values,"Checkpoint data round trip");
        std::cout<<"PASS: explicit shapes, allocation bounds, retained views and save/load\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
}
