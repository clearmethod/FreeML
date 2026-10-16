#include <objc/message.h>
#include <objc/runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <sstream>

#include <ToolsLibrary/Logger.h>

#if !defined(__APPLE__)
#error "This example requires Apple's Metal framework."
#endif

using Obj = void*;

struct MTLSize
{
    std::uint64_t width;
    std::uint64_t height;
    std::uint64_t depth;
};

template <typename Return, typename... Args>
Return Send(Obj receiver, const char* selectorName, Args... args)
{
    auto selector = sel_registerName(selectorName);
    auto fn = reinterpret_cast<Return (*)(Obj, SEL, Args...)>(objc_msgSend);
    return fn(receiver, selector, args...);
}

template <typename Return, typename... Args>
Return SendClass(const char* className, const char* selectorName, Args... args)
{
    return Send<Return>(reinterpret_cast<Obj>(objc_getClass(className)), selectorName, args...);
}

Obj NSString(const char* text)
{
    return SendClass<Obj>("NSString", "stringWithUTF8String:", text);
}

const char* UTF8String(Obj string)
{
    return Send<const char*>(string, "UTF8String");
}

void Release(Obj object)
{
    if (object)
    {
        Send<void>(object, "release");
    }
}

void PrintNSError(Obj error)
{
    if (!error)
    {
        return;
    }

    Obj description = Send<Obj>(error, "localizedDescription");
    std::cerr << UTF8String(description) << '\n';
}

int main()
{
    LOG_INFO() << "Starting MetalExample.";

    Obj autoreleasePool = SendClass<Obj>("NSAutoreleasePool", "new");

    void* metalFramework = dlopen(
        "/System/Library/Frameworks/Metal.framework/Metal",
        RTLD_LAZY | RTLD_LOCAL);
    if (!metalFramework)
    {
        std::cerr << "Failed to load Metal.framework\n";
        return 1;
    }

    using CreateDeviceFn = Obj (*)();
    auto createDevice = reinterpret_cast<CreateDeviceFn>(
        dlsym(metalFramework, "MTLCreateSystemDefaultDevice"));
    Obj device = createDevice ? createDevice() : nullptr;
    if (!device)
    {
        std::cerr << "No Metal device available\n";
        return 1;
    }
    LOG_INFO() << "Created Metal device.";

    constexpr const char* shaderSource = R"(
        #include <metal_stdlib>
        using namespace metal;

        kernel void add_one(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            uint id [[thread_position_in_grid]])
        {
            output[id] = input[id] + 1.0f;
        }
    )";

    Obj error = nullptr;
    Obj library = Send<Obj>(
        device,
        "newLibraryWithSource:options:error:",
        NSString(shaderSource),
        nullptr,
        &error);
    if (!library)
    {
        PrintNSError(error);
        return 1;
    }
    LOG_INFO() << "Compiled inline Metal shader library.";

    Obj function = Send<Obj>(library, "newFunctionWithName:", NSString("add_one"));
    Obj pipeline = Send<Obj>(
        device,
        "newComputePipelineStateWithFunction:error:",
        function,
        &error);
    if (!pipeline)
    {
        PrintNSError(error);
        return 1;
    }
    LOG_INFO() << "Created compute pipeline.";

    constexpr std::array<float, 4> inputValues = {1.0f, 2.0f, 3.0f, 4.0f};
    constexpr std::size_t byteCount = inputValues.size() * sizeof(float);
    constexpr std::uint64_t sharedStorage = 0;

    Obj input = Send<Obj>(
        device,
        "newBufferWithBytes:length:options:",
        inputValues.data(),
        byteCount,
        sharedStorage);
    Obj output = Send<Obj>(
        device,
        "newBufferWithLength:options:",
        byteCount,
        sharedStorage);

    Obj queue = Send<Obj>(device, "newCommandQueue");
    Obj commandBuffer = Send<Obj>(queue, "commandBuffer");
    Obj encoder = Send<Obj>(commandBuffer, "computeCommandEncoder");

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", input, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", output, 0ull, 1ull);

    MTLSize threads = {inputValues.size(), 1, 1};
    MTLSize threadsPerThreadgroup = {inputValues.size(), 1, 1};
    Send<void>(
        encoder,
        "dispatchThreads:threadsPerThreadgroup:",
        threads,
        threadsPerThreadgroup);

    Send<void>(encoder, "endEncoding");
    Send<void>(commandBuffer, "commit");
    Send<void>(commandBuffer, "waitUntilCompleted");
    LOG_INFO() << "Compute shader completed.";

    auto* result = Send<float*>(output, "contents");
    std::stringstream resultStream;
    for (std::size_t i = 0; i < inputValues.size(); ++i)
    {
        resultStream << result[i] << (i + 1 == inputValues.size() ? "" : " ");
    }
    LOG_INFO() << "Output: " << resultStream.str();

    Release(queue);
    Release(output);
    Release(input);
    Release(pipeline);
    Release(function);
    Release(library);
    Release(device);
    Send<void>(autoreleasePool, "drain");
    dlclose(metalFramework);

    return 0;
}
