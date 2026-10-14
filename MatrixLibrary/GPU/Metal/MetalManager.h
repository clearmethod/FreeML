#pragma once

#ifdef __APPLE__
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <objc/message.h>
#include <objc/runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cassert>
#include <filesystem>

#include <ToolsLibrary/Logger.h>


using Obj = void*;

constexpr uint64_t MTLResourceStorageModeShared = 0;
constexpr uint64_t MTLResourceStorageModePrivate = 2ull << 4;

struct MTLSize
{
    std::uint64_t width;
    std::uint64_t height;
    std::uint64_t depth;
};

struct NSRange
{
    std::uint64_t location;
    std::uint64_t length;
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

class MetalManager
{
    public:
    static bool IsAlive()
    {
        return LifetimeState().alive;
    }

    static MetalManager* Instance()
    {
        static MetalManager instance;
        instance.Init();
        return &instance;
    }

    void WorkSubmitted()
    {
        m_workToCommit = true;
    }

    bool Init()
    {
        if(!m_init)
        {
			LOG_INFO() << "Initializing MetalManager";
            // Create the D3D11 device and immediate context.
            m_metalFramework = dlopen("/System/Library/Frameworks/Metal.framework/Metal",
                                        RTLD_LAZY | RTLD_LOCAL);
            if (!m_metalFramework)
            {
                std::cerr << "Failed to load Metal.framework\n";
                return false;
            }

            using CreateDeviceFn = Obj (*)();
            auto createDevice = reinterpret_cast<CreateDeviceFn>(
                dlsym(m_metalFramework, "MTLCreateSystemDefaultDevice"));
            m_device = createDevice ? createDevice() : nullptr;
            if (!m_device)
            {
                std::cerr << "No Metal device available\n";
                return false;
            }

            m_queue           = Send<Obj>(m_device, "newCommandQueue");
            m_commandBuffer   = Send<Obj>(m_queue, "commandBuffer");
            m_encoder         = Send<Obj>(m_commandBuffer, "computeCommandEncoder");

            LOG_INFO() << "Loading Base Shaders.";
            LOG_INFO() << "\t from Current working dir:" << std::filesystem::current_path();

            //Load shaders
            LoadComputeShaderFromFile("GPU/Metal/Shaders/mul.metal", "mul");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/mul_tiled.metal", "mul_tiled");
            // SIMD-group matrix operations require Apple GPU family 7 or later.
            if (Send<bool>(m_device, "supportsFamily:", 1007ull))
                LoadComputeShaderFromFile("GPU/Metal/Shaders/mul_simdgroup.metal", "mul_simdgroup");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/mul_transposeleft.metal", "mul_transposeleft");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/mul_transposeright.metal", "mul_transposeright");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/mul_transposeboth.metal", "mul_transposeboth");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/mul_strided.metal", "mul_strided");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/mul_strided_transposeleft.metal", "mul_strided_transposeleft");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/mul_strided_transposeright.metal", "mul_strided_transposeright");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/mul_strided_transposeboth.metal", "mul_strided_transposeboth");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/add.metal", "add");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/add_inplace.metal", "add_inplace");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/sub.metal", "sub");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/sub_inplace.metal", "sub_inplace");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/broadcast_add.metal", "broadcast_add");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/split_qkv.metal", "split_qkv");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/merge_qkv.metal", "merge_qkv");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/scatter_add_rows.metal", "scatter_add_rows");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/gather_rows.metal", "gather_rows");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/per_element_mul.metal", "per_element_mul");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/per_element_mul_inplace.metal", "per_element_mul_inplace");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/scale.metal", "scale");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/scale_inplace.metal", "scale_inplace");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/scale_scalar.metal", "scale_scalar");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/scale_scalar_inplace.metal", "scale_scalar_inplace");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/fill.metal", "fill");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/fill_scalar.metal", "fill_scalar");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/sum_columns.metal", "sum_columns");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/gelu.metal", "gelu");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/gelu_inplace.metal", "gelu_inplace");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/gelu_derivative.metal", "gelu_derivative");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/gelu_backwards.metal", "gelu_backwards");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/gelu_derivative_inplace.metal", "gelu_derivative_inplace");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/scaleadd.metal", "scaleadd");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/scaleadd_inplace.metal", "scaleadd_inplace");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/softmax.metal", "softmax");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/cce_logits.metal", "cce_logits_count");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/cce_logits.metal", "cce_logits_grad");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/cce_logits.metal", "cce_logits_loss");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/cce_logits.metal", "cce_logits_reduce");

            LoadComputeShaderFromFile("GPU/Metal/Shaders/softmax_backwards.metal", "softmax_backwards");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/attention_scores_backwards.metal", "attention_scores_backwards");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/adam_update.metal", "adam_update");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/layernorm.metal", "layernorm");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/layernorm_cached.metal", "layernorm_cached");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/layernorm_nobeta_cached.metal", "layernorm_nobeta_cached");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/layernorm_backwards_dx_cached.metal", "layernorm_backwards_dx_cached");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/layernorm_nobeta.metal", "layernorm_nobeta");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/layernorm_no_xhat.metal", "layernorm_no_xhat");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/layernorm_nobeta_no_xhat.metal", "layernorm_nobeta_no_xhat");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/layernorm_backwards_dx.metal", "layernorm_backwards_dx");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/layernorm_backwards_params.metal", "layernorm_backwards_params");
            LoadComputeShaderFromFile("GPU/Metal/Shaders/layernorm_backwards_params_nobeta.metal", "layernorm_backwards_params_nobeta");

            // Prealloc cbuffers
            if (m_cbufferFreeList.empty())
            {
                constexpr uint64_t reserveCbuffCount = 1024u;
                m_cbufferFreeList.reserve(reserveCbuffCount);
                for (uint32_t i = 0; i < reserveCbuffCount; ++i)
                {
                    int32_t handle = CreateAndPopulateCBuffer(0u, 0u, 0u, 0u, 0u);
                    if (handle >= 0)
                    {
                        m_cbufferFreeList.push_back(handle);
                    }
                }
            }

            m_init = true;
        }
        return true;
    }

    // Submit in queue order without blocking the CPU. Use WaitForGPU() when
    // completion is required, including when all work is already submitted.
    void SubmitRecordedWork()
    {
        if(m_workToCommit)
        {
            if (m_encoder) Send<void>(m_encoder, "endEncoding");
            CommitWithoutWaiting(m_commandBuffer);
            m_workToCommit = false;

            // Get new next command buffer and encoder
            m_commandBuffer   = Send<Obj>(m_queue, "commandBuffer");
            m_encoder         = Send<Obj>(m_commandBuffer, "computeCommandEncoder");
        }
    }

    int32_t CreateBuffer(uint32_t sizeBytes, uint32_t elementCount)
    {
        Obj gpuBuffer = Send<Obj>(
            m_device,
            "newBufferWithLength:options:",
            sizeBytes,
            MTLResourceStorageModePrivate);

        Obj stagingBuffer = Send<Obj>(
            m_device,
            "newBufferWithLength:options:",
            sizeBytes,
            MTLResourceStorageModeShared);

        m_bufferHandles++;
        m_bufferMap[m_bufferHandles]        = gpuBuffer;
        m_bufferStagingMap[m_bufferHandles] = stagingBuffer;
		m_bufferSizesMap[m_bufferHandles]   = sizeBytes;
		m_allocatedBytes += sizeBytes;

		return m_bufferHandles;
    }

    struct alignas(16) CBufferParams
    {
        uint32_t x;
        uint32_t y;
        uint32_t z;
        uint32_t offset;
        uint32_t uniqueId;
        uint32_t pad[3];
        // Matches HLSL layout: uint4 optionalParams[4].
        uint32_t optionalParams[16];
    };
    static_assert(sizeof(CBufferParams) == 96, "CBufferParams size must match HLSL cbuffer layout.");

    int32_t CreateAndPopulateCBuffer(uint32_t x, uint32_t y, uint32_t z, uint32_t offset, uint32_t uniqueId)
    {
        if (!m_device)
        {
            return -1;
        }

        CBufferParams params = {};
        params.x = x;
        params.y = y;
        params.z = z;
        params.offset = offset;
        params.uniqueId = uniqueId;

        Obj cbuffer = Send<Obj>(
            m_device,
            "newBufferWithBytes:length:options:",
            &params,
            sizeof(params),
            MTLResourceStorageModeShared);
        if (!cbuffer)
        {
            return -1;
        }

        m_cbufferHandles++;
        m_cbufferMap[m_cbufferHandles] = cbuffer;
        m_cbufferParamsMap[m_cbufferHandles] = params;
        return m_cbufferHandles;
    }

    bool UpdateCBuffer(int32_t handle, uint32_t x, uint32_t y, uint32_t z, uint32_t offset, uint32_t uniqueId)
    {
        auto it = m_cbufferMap.find(handle);
        if (it == m_cbufferMap.end())
        {
            return false;
        }

        void* ptr = Send<void*>(it->second, "contents");
        if (!ptr)
        {
            return false;
        }

        CBufferParams params = {};
        params.x = x;
        params.y = y;
        params.z = z;
        params.offset = offset;
        params.uniqueId = uniqueId;

        auto itParams = m_cbufferParamsMap.find(handle);
        if (itParams != m_cbufferParamsMap.end())
        {
            std::memcpy(params.optionalParams, itParams->second.optionalParams, sizeof(params.optionalParams));
        }

        std::memcpy(ptr, &params, sizeof(params));
        m_cbufferParamsMap[handle] = params;
        return true;
    }

    int32_t AcquireCachedCBuffer(uint32_t x, uint32_t y, uint32_t z, uint32_t offset, uint32_t uniqueId)
    {
        int32_t handle = -1;
        if (!m_cbufferFreeList.empty())
        {
            handle = m_cbufferFreeList.back();
            m_cbufferFreeList.pop_back();
        }
        else
        {
			LOG_INFO() << "CBuffer free list empty, creating new CBuffer.";
            handle = CreateAndPopulateCBuffer(x, y, z, offset, uniqueId);
            return handle;
        }

        if (!UpdateCBuffer(handle, x, y, z, offset, uniqueId))
        {
            return -1;
        }

        return handle;
    }

    bool UpdateCachedCBuffer(int32_t handle, uint32_t x, uint32_t y, uint32_t z, uint32_t offset, uint32_t uniqueId)
    {
        return UpdateCBuffer(handle, x, y, z, offset, uniqueId);
    }

    bool ReleaseCachedCBuffer(int32_t handle)
    {
        if (handle < 0)
        {
            return false;
        }

        auto itParams = m_cbufferParamsMap.find(handle);
        if (itParams != m_cbufferParamsMap.end())
        {
            std::memset(itParams->second.optionalParams, 0, sizeof(itParams->second.optionalParams));
        }
        m_cbufferFreeList.push_back(handle);
        return true;
    }

    bool DeleteCBuffer(int32_t handle)
    {
        auto it = m_cbufferMap.find(handle);
        if (it == m_cbufferMap.end())
        {
            return false;
        }

        Release(it->second);
        m_cbufferMap.erase(it);
        m_cbufferParamsMap.erase(handle);
        return true;
    }

    bool DeleteBuffer(int32_t handle)
    {
        if (!IsAlive())
        {
            return false;
        }

        auto itBuffer = m_bufferMap.find(handle);
        if (itBuffer == m_bufferMap.end())
        {
            return false;
        }

        Release(itBuffer->second);
        m_bufferMap.erase(itBuffer);

        auto itStaging = m_bufferStagingMap.find(handle);
        if (itStaging != m_bufferStagingMap.end())
        {
            Release(itStaging->second);
            m_bufferStagingMap.erase(itStaging);
        }

        m_bufferMappedPtrMap.erase(handle);
        m_bufferDirtyMappings.erase(handle);
        m_bufferStagingMap.erase(handle);

        auto itSize = m_bufferSizesMap.find(handle);
        if (itSize != m_bufferSizesMap.end())
        {
            m_allocatedBytes -= itSize->second;
            m_bufferSizesMap.erase(itSize);
        }

        return true;

    }

    std::string GetMemoryString()
    {
        const double allocatedMB = static_cast<double>(m_allocatedBytes) / (1024.0 * 1024.0);
        std::stringstream ss;
        ss << "GPU Buffer Allocated Bytes: " << m_allocatedBytes << " bytes (" << allocatedMB << " MB)";
        return ss.str();
    }

    std::string GetString()
    {
		std::stringstream ss;
		ss << "MetalManager Report:" << std::endl;
		//ss << "  Allocated Buffers: " << m_bufferMap.size() << std::endl;        
        ss << "  " << GetMemoryString() << std::endl;
			ss << "  Cached CBuffers: " << (m_cbufferMap.size() - m_cbufferFreeList.size()) << std::endl;
			ss << "  Total CBuffers: " << m_cbufferMap.size() << std::endl;
		//ss << "  Total Shaders: " << m_shaderMap.size() << std::endl;
        // Loop around and print each shader
        for(auto shaderPair : m_shaderPipelinesMap)
        {
            ss << "    Shader: " << shaderPair.first << std::endl;
		}
		return ss.str();
    }

    uint64_t GetAllocatedBytes() const
    {
        return m_allocatedBytes;
	}

    Obj GetBuffer(int32_t handle) const
    {
        auto it = m_bufferMap.find(handle);
        if (it == m_bufferMap.end())
        {
            return nullptr;
        }
        return it->second;
    }

    Obj GetQueue() const
    {
        return m_queue;
    }

    Obj GetCommandBuffer() const
    {
        return m_commandBuffer;
    }

    Obj GetEncoder() const
    {
        return m_encoder;
    }

    Obj GetStagingBuffer(int32_t handle) const
    {
        auto it = m_bufferStagingMap.find(handle);
        if (it == m_bufferStagingMap.end())
        {
            return nullptr;
        }
        return it->second;
    }

    void* GetMappedPointer(int32_t handle) const
    {
        auto it = m_bufferMappedPtrMap.find(handle);
        if (it == m_bufferMappedPtrMap.end())
        {
            return nullptr;
        }

        return it->second;
    }

    bool IsMappingDirty(int32_t handle) const
    {
        return m_bufferDirtyMappings.contains(handle);
    }

    void MarkMappingDirty(int32_t handle)
    {
        m_bufferDirtyMappings.insert(handle);
    }

    void* MapBuffer(int32_t handle, bool writable = true)
    {
        Obj stagingBuffer = GetStagingBuffer(handle);
        if (!stagingBuffer)
        {
            return nullptr;
        }

        void* ptr = Send<void*>(stagingBuffer, "contents");
        if (!ptr)
        {
            return nullptr;
        }

        m_bufferMappedPtrMap[handle] = ptr;
        if (writable)
            MarkMappingDirty(handle);
        return ptr;
    }

    void UnmapBuffer(int32_t handle)
    {
        m_bufferMappedPtrMap.erase(handle);
        m_bufferDirtyMappings.erase(handle);
    }
   
    bool CopyStagingToGPU(int32_t handle)
    {
        Obj gpuBuffer     = GetBuffer(handle);
        Obj stagingBuffer = GetStagingBuffer(handle);
        if (!gpuBuffer || !stagingBuffer)
        {
            return false;
        }

        // A separately committed blit must not overtake recorded compute work.
        SubmitRecordedWork();
        Obj commandBuffer = Send<Obj>(m_queue, "commandBuffer");
        Obj blit = Send<Obj>(commandBuffer, "blitCommandEncoder");

        uint32_t sizeBytes = m_bufferSizesMap[handle];

        Send<void>(
            blit,
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
            stagingBuffer,
            0ull,
            gpuBuffer,
            0ull,
            sizeBytes);

        Send<void>(blit, "endEncoding");
        CommitWithoutWaiting(commandBuffer);
        // The caller may immediately overwrite the reusable staging memory.
        WaitForGPU();
        return true;
    }

    bool CopyGPUToStaging(int32_t handle)
    {
        Obj gpuBuffer     = GetBuffer(handle);
        Obj stagingBuffer = GetStagingBuffer(handle);
        if (!gpuBuffer || !stagingBuffer)
        {
            return false;
        }

        SubmitRecordedWork();
        Obj commandBuffer = Send<Obj>(m_queue, "commandBuffer");
        Obj blit          = Send<Obj>(commandBuffer, "blitCommandEncoder");

        uint32_t sizeBytes = m_bufferSizesMap[handle];

        Send<void>( blit,
                    "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
                    gpuBuffer,
                    0ull,
                    stagingBuffer,
                    0ull,
                    sizeBytes);

        Send<void>(blit, "endEncoding");
        CommitWithoutWaiting(commandBuffer);
        // CPU reads must see the completed download.
        WaitForGPU();

        return true;
    }

    bool CopyBuffer(int32_t outHandle, int32_t inHandle,
                    uint64_t outOffset = 0, uint64_t inOffset = 0,
                    uint64_t size = UINT64_MAX)
    {
        if (outHandle == inHandle && outOffset == inOffset)
        {
            return true;
        }

        Obj outBuffer = GetBuffer(outHandle);
        Obj inBuffer = GetBuffer(inHandle);
        if (!outBuffer || !inBuffer)
        {
            return false;
        }

        auto outSizeIt = m_bufferSizesMap.find(outHandle);
        auto inSizeIt = m_bufferSizesMap.find(inHandle);
        if (outSizeIt == m_bufferSizesMap.end() || inSizeIt == m_bufferSizesMap.end())
        {
            return false;
        }

        if (size == UINT64_MAX)
            size = outSizeIt->second;
        if (outOffset > outSizeIt->second || inOffset > inSizeIt->second
            || size > outSizeIt->second - outOffset || size > inSizeIt->second - inOffset)
        {
            LOG_INFO() << "CopyBuffer failed: copy range exceeds source or destination buffer.";
            return false;
        }

        if (size == 0u) return true;
        if (!m_commandBuffer || !m_encoder) return false;

        // Switch encoders without submitting: earlier compute, this copy and
        // subsequent compute execute in order within the same command buffer.
        Send<void>(m_encoder, "endEncoding");
        m_encoder = nullptr;
        Obj blit = Send<Obj>(m_commandBuffer, "blitCommandEncoder");
        if (!blit)
        {
            m_encoder = Send<Obj>(m_commandBuffer, "computeCommandEncoder");
            return false;
        }

        Send<void>(
            blit,
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
            inBuffer,
            inOffset,
            outBuffer,
            outOffset,
            size);

        Send<void>(blit, "endEncoding");
        WorkSubmitted(); // Copy-only buffers must also be submitted by WaitForGPU.
        m_encoder = Send<Obj>(m_commandBuffer, "computeCommandEncoder");
        if (!m_encoder)
        {
            LOG_INFO() << "CopyBuffer failed to resume compute encoding.";
            return false;
        }

        return true;
    }

    bool ClearBuffer(int32_t handle, uint32_t elementOffset = 0u,
                     uint32_t elementCount = UINT32_MAX)
    {
        Obj buffer = GetBuffer(handle);
        Obj pipeline = GetPipeline("fill_scalar");
        auto sizeIt = m_bufferSizesMap.find(handle);
        if (!buffer || !pipeline || !m_encoder || sizeIt == m_bufferSizesMap.end())
            return false;

        // MatrixMetal stores float elements in contiguous, row-major buffers.
        const uint64_t capacity = sizeIt->second / sizeof(float);
        if (elementOffset > capacity)
            return false;
        if (elementCount == UINT32_MAX)
            elementCount = static_cast<uint32_t>(capacity - elementOffset);
        if (uint64_t(elementOffset) + elementCount > capacity)
            return false;
        if (elementCount == 0u)
            return true;

        const bool wholeBuffer = elementOffset == 0u && elementCount == capacity;
        if (GetMappedPointer(handle))
        {
            // A full clear overwrites all CPU edits. A slice clear must first
            // preserve edits elsewhere in the shared backing buffer.
            if (!wholeBuffer && IsMappingDirty(handle) && !CopyStagingToGPU(handle))
                return false;
            UnmapBuffer(handle);
        }

        CBufferParams params = {};
        params.x = elementCount;
        params.y = params.z = 1u;
        params.offset = elementOffset;
        // optionalParams[0] remains the float bit pattern for zero.
        Send<void>(m_encoder, "setComputePipelineState:", pipeline);
        Send<void>(m_encoder, "setBuffer:offset:atIndex:", buffer, 0ull, 0ull);
        Send<void>(m_encoder, "setBytes:length:atIndex:", &params, sizeof(params), 1ull);
        MTLSize threads = {elementCount, 1u, 1u};
        MTLSize group = {256u, 1u, 1u};
        Send<void>(m_encoder, "dispatchThreads:threadsPerThreadgroup:", threads, group);
        WorkSubmitted();
        return true;
    }

    Obj GetCBuffer(int32_t handle) const
    {
        auto it = m_cbufferMap.find(handle);
        if (it == m_cbufferMap.end())
        {
            return nullptr;
        }

        return it->second;
    }

    Obj GetPipeline(const std::string& name) const
    {
        auto it = m_shaderPipelinesMap.find(name);
        if (it == m_shaderPipelinesMap.end())
        {
            return nullptr;
        }

        return it->second;
    }

    bool SetCBufferOptionalParams(int32_t handle, const uint32_t* optionalParams, uint32_t optionalCount)
    {
        auto it = m_cbufferMap.find(handle);
        if (it == m_cbufferMap.end())
        {
            LOG_INFO() << "SetCBufferOptionalParams failed: cbuffer handle not found: " << handle;
            return false;
        }

        auto itParams = m_cbufferParamsMap.find(handle);
        if (itParams == m_cbufferParamsMap.end())
        {
            LOG_INFO() << "SetCBufferOptionalParams failed: params not found for handle: " << handle;
            return false;
        }

        void* ptr = Send<void*>(it->second, "contents");
        if (!ptr)
        {
            LOG_INFO() << "SetCBufferOptionalParams failed: contents() returned null for handle: " << handle;
            return false;
        }

        CBufferParams params = itParams->second;
        std::memset(params.optionalParams, 0, sizeof(params.optionalParams));
        if (optionalParams && optionalCount > 0u)
        {
            const uint32_t count = optionalCount < 16u ? optionalCount : 16u;
            std::memcpy(params.optionalParams, optionalParams, sizeof(uint32_t) * count);
        }

        std::memcpy(ptr, &params, sizeof(params));
        m_cbufferParamsMap[handle] = params;
        return true;
    }

    bool GetCBufferOptionalParams(int32_t handle, uint32_t* outParams, uint32_t outCount) const
    {
        if (!outParams || outCount == 0u)
        {
            return false;
        }

        auto it = m_cbufferParamsMap.find(handle);
        if (it == m_cbufferParamsMap.end())
        {
            return false;
        }

        const uint32_t count = outCount < 16u ? outCount : 16u;
        std::memcpy(outParams, it->second.optionalParams, sizeof(uint32_t) * count);
        return true;
    }

    bool GetCBufferParams(int32_t handle, CBufferParams* outParams) const
    {
        if (!outParams)
        {
            return false;
        }

        auto it = m_cbufferParamsMap.find(handle);
        if (it == m_cbufferParamsMap.end())
        {
            return false;
        }

        *outParams = it->second;
        return true;
    }

    bool WaitForGPU()
    {
        SubmitRecordedWork();
        if (m_lastSubmitted)
        {
            // All work uses one queue in commit order. Completing its latest
            // command buffer also completes preceding compute and blits.
            Send<void>(m_lastSubmitted, "waitUntilCompleted");
            Release(m_lastSubmitted);
            m_lastSubmitted = nullptr;
        }
        return true;
    }

    bool LoadComputeShaderFromFile(std::string _filepath, std::string _name)
    {
        std::string computeShaderString;
        std::ifstream file(_filepath, std::ios::binary | std::ios::ate);
        if (!file)
        {
            LOG_INFO() << "Failed to find compute shader file: " << _filepath;
            assert(false);
            return false;
        }

        std::ifstream::pos_type size = file.tellg();
        if (size <= 0)
        {
            LOG_INFO() << "Compute shader file is empty: " << _filepath;
            assert(false);
            return false;
        }

        computeShaderString.resize(static_cast<size_t>(size));
        file.seekg(0, std::ios::beg);
        if (!file.read(&computeShaderString[0], size))
        {
            LOG_INFO() << "Failed to read compute shader file: " << _filepath;
            assert(false);
            return false;
        }

        return LoadComputeShader(computeShaderString.c_str(), _name);
    }

    bool LoadComputeShader(const char* _shader, std::string _name)
    {
        Obj error = nullptr;
        Obj library = Send<Obj>(m_device,
                                "newLibraryWithSource:options:error:",
                                NSString(_shader),
                                nullptr,
                                &error);
        Obj function = Send<Obj>(library, "newFunctionWithName:", NSString(_name.c_str()));
        Obj pipeline = Send<Obj>(m_device,
                                 "newComputePipelineStateWithFunction:error:",
                                 function,
                                 &error);
        if (!pipeline)
        {
            PrintNSError(error);
            return 1;
        }

        m_shaderPipelines.push_back(pipeline);
        m_shaderPipelinesMap[_name] = pipeline;
        m_shaderLibraryMap[_name] = library;

        return true;
    }

    void DumpInfoQueueErrors()
    {
    }

    private:
    struct LifetimeFlag
    {
        bool alive = true;
    };

    static LifetimeFlag& LifetimeState()
    {
        static LifetimeFlag* state = new LifetimeFlag();
        return *state;
    }

    MetalManager() = default;
    ~MetalManager()
    {
        WaitForGPU();
        LifetimeState().alive = false;
        m_init = false;

        // Delete objc metal objects here
    }

    public:

    std::vector<Obj>             m_shaderPipelines;
    std::map<std::string, Obj>   m_shaderPipelinesMap;
    std::map<std::string, Obj>   m_shaderLibraryMap;

    int32_t                           m_bufferHandles = -1;
    std::map<int32_t, Obj>            m_bufferMap;
    std::map<int32_t, uint64_t>       m_bufferSizesMap;
    std::map<int32_t, Obj>            m_bufferStagingMap;
    std::map<int32_t, void*>          m_bufferMappedPtrMap;
    // Mapping state belongs to the backing buffer, including all slice views.
    std::set<int32_t>                m_bufferDirtyMappings;

    int32_t                           m_cbufferHandles = -1;
    std::map<int32_t, Obj>            m_cbufferMap;
    std::map<int32_t, CBufferParams>  m_cbufferParamsMap;
    std::vector<int32_t>              m_cbufferFreeList;

    uint64_t m_allocatedBytes;

    bool m_init = false;

    void* m_metalFramework = nullptr;
    Obj   m_device  = nullptr;
    Obj   m_library = nullptr;
    Obj   m_queue   = nullptr;
    Obj   m_encoder = nullptr;
    Obj   m_commandBuffer = nullptr;
    Obj   m_lastSubmitted = nullptr;

    bool m_workToCommit = false;

private:
    void CommitWithoutWaiting(Obj commandBuffer)
    {
        // commandBuffer is autoreleased; retain the completion boundary even
        // when no further compute work is recorded before WaitForGPU().
        Send<void>(commandBuffer, "retain");
        Send<void>(commandBuffer, "commit");
        Release(m_lastSubmitted);
        m_lastSubmitted = commandBuffer;
    }
};


#endif
