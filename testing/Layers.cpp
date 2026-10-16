#include "LayerTestSupport.h"

using namespace LayerTesting;

int main(int argc,char** argv)
{
    if(argc!=3) { std::cerr<<"Usage: LayerTests_<backend> <layer> <Forward|Backward>\n"; return 2; }
    try {
        // Initialize logging before managers, which log during static teardown.
        LOG_INFO() << "Starting layer test";
        Require(std::find(cases.begin(),cases.end(),argv[1])!=cases.end(),"Unknown layer");
#if defined(LAYER_TEST_Metal)
        if(!MetalManager::Instance()->GetQueue()) { std::cout<<"SKIP: No Metal device available\n"; return 77; }
#elif defined(LAYER_TEST_DX11)
        if(!DirectX11Manager::Instance()->GetContext()) { std::cout<<"SKIP: No DirectX 11 device available\n"; return 77; }
#endif
        Fixture<TestMatrix> fixture(argv[1]);
        if(std::string(argv[2])=="Forward") {
            CheckForward(fixture);
            if(fixture.name.starts_with("LayerNorm")) CheckLayerNormCacheLifecycle<TestMatrix>(fixture.name);
        }
        else if(std::string(argv[2])=="Backward") CheckBackward(fixture);
        else throw std::runtime_error("Unknown pass");
        std::cout<<"PASS: "<<argv[1]<<' '<<argv[2]<<'\n'; return 0;
    } catch(const std::exception& error) { std::cerr<<"FAIL: "<<argv[1]<<' '<<argv[2]<<": "<<error.what()<<'\n'; return 1; }
}
