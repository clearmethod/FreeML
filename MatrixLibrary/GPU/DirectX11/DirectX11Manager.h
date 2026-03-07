#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <filesystem>

#include <ToolsLibrary/Logger.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

class DirectX11Manager
{
    public:
    static bool IsAlive()
    {
        return LifetimeState().alive;
    }

    static DirectX11Manager* Instance()
    {
        static DirectX11Manager instance;
        instance.Init();
        return &instance;
    }

    ID3D11DeviceContext* GetContext()
    {
        return m_context;
    }


    bool Init()
    {
        if(!m_init)
        {
			LOG_INFO() << "Initializing DirectX11Manager";
            // Create the D3D11 device and immediate context.
            D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
            D3D_FEATURE_LEVEL requested[] = { D3D_FEATURE_LEVEL_11_0 };
            UINT flags = 0u;
            #if defined(_DEBUG)
                flags |= D3D11_CREATE_DEVICE_DEBUG;
            #endif

            HRESULT hr = D3D11CreateDevice(nullptr,
                                        D3D_DRIVER_TYPE_HARDWARE,
                                        nullptr,
                                        flags,
                                        requested,
                                        1u,
                                        D3D11_SDK_VERSION,
                                        &m_device,
                                        &featureLevel,
                                        &m_context);
            if (FAILED(hr))
            {
				LOG_ERROR() << "Failed to create D3D11 device.";
                return false;
            }


            LOG_INFO() << "Loading Base Shaders.";
            LOG_INFO() << "Current working dir:" << std::filesystem::current_path();

            //Load shaders
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/mul.hlsl", "mul");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/mul_transposeleft.hlsl", "mul_transposeleft");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/mul_transposeright.hlsl", "mul_transposeright");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/mul_transposeboth.hlsl", "mul_transposeboth");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/mul_strided.hlsl", "mul_strided");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/mul_strided_transposeleft.hlsl", "mul_strided_transposeleft");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/mul_strided_transposeright.hlsl", "mul_strided_transposeright");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/mul_strided_transposeboth.hlsl", "mul_strided_transposeboth");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/add.hlsl", "add");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/add_inplace.hlsl", "add_inplace");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/sub.hlsl", "sub");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/sub_inplace.hlsl", "sub_inplace");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/broadcast_add.hlsl", "broadcast_add");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/scaleadd.hlsl", "scaleadd");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/scaleadd_inplace.hlsl", "scaleadd_inplace");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/per_element_mul.hlsl", "per_element_mul");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/per_element_mul_inplace.hlsl", "per_element_mul_inplace");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/copy_range.hlsl", "copy_range");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/gather_rows.hlsl", "gather_rows");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/scatter_rows.hlsl", "scatter_rows");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/scatter_add_row.hlsl", "scatter_add_row");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/split_qkv.hlsl", "split_qkv");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/merge_qkv.hlsl", "merge_qkv");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/sum_spatial_dimension.hlsl", "sum_spatial_dimension");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/conv2d_single_channel.hlsl", "conv2d_single_channel");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/conv2d_single_channel_bias.hlsl", "conv2d_single_channel_bias");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/conv2d_single_channel_backwards_wupdate.hlsl", "conv2d_single_channel_backwards_wupdate");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/conv2d_single_channel_backwards_errorout.hlsl", "conv2d_single_channel_backwards_errorout");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/conv2d_transpose_single_channel.hlsl", "conv2d_transpose_single_channel");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/conv2d_transpose_single_channel_bias.hlsl", "conv2d_transpose_single_channel_bias");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/conv2d_transpose_single_channel_backwards_wupdate.hlsl", "conv2d_transpose_single_channel_backwards_wupdate");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/conv2d_transpose_single_channel_backwards_errorout.hlsl", "conv2d_transpose_single_channel_backwards_errorout");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/relu.hlsl", "relu");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/relu_inplace.hlsl", "relu_inplace");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/relu_derivative.hlsl", "relu_derivative");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/relu_derivative_inplace.hlsl", "relu_derivative_inplace");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/gelu.hlsl", "gelu");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/gelu_inplace.hlsl", "gelu_inplace");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/gelu_derivative.hlsl", "gelu_derivative");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/gelu_derivative_inplace.hlsl", "gelu_derivative_inplace");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/transpose.hlsl", "transpose");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/scale.hlsl", "scale");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/scale_inplace.hlsl", "scale_inplace");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/scale_scalar.hlsl", "scale_scalar");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/scale_scalar_inplace.hlsl", "scale_scalar_inplace");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/fill.hlsl", "fill");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/fill_scalar.hlsl", "fill_scalar");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/softmax.hlsl", "softmax");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/softmax_backwards.hlsl", "softmax_backwards");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/cce_logits_loss.hlsl", "cce_logits_loss");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/cce_logits_grad.hlsl", "cce_logits_grad");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/mse_loss_reduce_rows.hlsl", "mse_loss_reduce_rows");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/reduce_sum_to_scalar.hlsl", "reduce_sum_to_scalar");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/adam_update.hlsl", "adam_update");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/layernorm.hlsl", "layernorm");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/layernorm_nobeta.hlsl", "layernorm_nobeta");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/layernorm_no_xhat.hlsl", "layernorm_no_xhat");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/layernorm_nobeta_no_xhat.hlsl", "layernorm_nobeta_no_xhat");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/layernorm_backwards_dx.hlsl", "layernorm_backwards_dx");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/layernorm_backwards_params.hlsl", "layernorm_backwards_params");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/layernorm_backwards_params_nobeta.hlsl", "layernorm_backwards_params_nobeta");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/reparameterize.hlsl", "reparameterize");
            LoadComputeShaderFromFile("GPU/DirectX11/Shaders/reparameterize_backwards.hlsl", "reparameterize_backwards");

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

    int32_t CreateBuffer(uint32_t sizeBytes, uint32_t elementCount)
    {
		LOG_INFO() << "Creating GPU buffer of size (bytes): " << sizeBytes << " Elements: " << elementCount;

        D3D11_BUFFER_DESC gpuDesc = {};
        gpuDesc.ByteWidth = sizeBytes;
        gpuDesc.Usage = D3D11_USAGE_DEFAULT;
        gpuDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        gpuDesc.CPUAccessFlags = 0u;
        gpuDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        gpuDesc.StructureByteStride = sizeof(float);

        ComPtr<ID3D11Buffer> gpuBuffer;
        HRESULT hr = m_device->CreateBuffer(&gpuDesc, nullptr, &gpuBuffer);
        if (FAILED(hr))
        {
            return -1;
        }

        D3D11_BUFFER_DESC stagingDesc = {};
        stagingDesc.ByteWidth = sizeBytes;
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.BindFlags = 0u;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
        stagingDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        stagingDesc.StructureByteStride = sizeof(float);

        ComPtr<ID3D11Buffer> stagingBuffer;
        hr = m_device->CreateBuffer(&stagingDesc, nullptr, &stagingBuffer);
        if (FAILED(hr))
        {
            return -1;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = elementCount;

        ComPtr<ID3D11ShaderResourceView> inputSRV;
        hr = m_device->CreateShaderResourceView(gpuBuffer.Get(), &srvDesc, &inputSRV);
        if (FAILED(hr))
        {
            // TODO destroy buffer object.
            return -1;
        }

        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format              = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.NumElements  = elementCount;

        ComPtr<ID3D11UnorderedAccessView> inputUAV;
        hr = m_device->CreateUnorderedAccessView(gpuBuffer.Get(), &uavDesc, &inputUAV);
        if (FAILED(hr))
        {
            // TODO destroy buffer object.
            return -1;
        }

        m_bufferHandles++;
        m_bufferMap[m_bufferHandles]        = gpuBuffer;
        m_bufferStagingMap[m_bufferHandles] = stagingBuffer;
        m_bufferSRVMap[m_bufferHandles]     = inputSRV;
        m_bufferUAVMap[m_bufferHandles]     = inputUAV;
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
        //LOG_INFO() << "Creating and populating CBuffer";

        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = sizeof(CBufferParams);
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        ComPtr<ID3D11Buffer> cbuffer;
        HRESULT hr = m_device->CreateBuffer(&desc, nullptr, &cbuffer);
        if (FAILED(hr))
        {
            return -1;
        }

        D3D11_MAPPED_SUBRESOURCE mapped = {};
        hr = m_context->Map(cbuffer.Get(), 0u, D3D11_MAP_WRITE_DISCARD, 0u, &mapped);
        if (FAILED(hr))
        {
            return -1;
        }

        CBufferParams params = {};
        params.x = x;
        params.y = y;
        params.z = z;
        params.offset = offset;
        params.uniqueId = uniqueId;
        std::memcpy(mapped.pData, &params, sizeof(params));
        m_context->Unmap(cbuffer.Get(), 0u);

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

        D3D11_MAPPED_SUBRESOURCE mapped = {};
        HRESULT hr = m_context->Map(it->second.Get(), 0u, D3D11_MAP_WRITE_DISCARD, 0u, &mapped);
        if (FAILED(hr))
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
        std::memcpy(mapped.pData, &params, sizeof(params));
        m_context->Unmap(it->second.Get(), 0u);
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

        m_bufferMap.erase(itBuffer);
        m_bufferStagingMap.erase(handle);
        m_bufferSRVMap.erase(handle);
        m_bufferUAVMap.erase(handle);

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
		ss << "DirectX11Manager Report:" << std::endl;
		ss << "  Allocated Buffers: " << m_bufferMap.size() << std::endl;        
        ss << "  " << GetMemoryString() << std::endl;
		ss << "  Cached CBuffers: " << (m_cbufferMap.size() - m_cbufferFreeList.size()) << std::endl;
		ss << "  Total CBuffers: " << m_cbufferMap.size() << std::endl;
		ss << "  Total Shaders: " << m_shaderMap.size() << std::endl;
        // Loop around and print each shader
        for(auto shaderPair : m_shaderMap)
        {
            ss << "    Shader: " << shaderPair.first << std::endl;
		}
		return ss.str();
    }

    uint64_t GetAllocatedBytes() const
    {
        return m_allocatedBytes;
	}

    ID3D11Buffer* GetBuffer(int32_t handle) const
    {
        auto it = m_bufferMap.find(handle);
        if (it == m_bufferMap.end())
        {
            return nullptr;
        }

        return it->second.Get();
    }

    ID3D11Buffer* GetStagingBuffer(int32_t handle) const
    {
        auto it = m_bufferStagingMap.find(handle);
        if (it == m_bufferStagingMap.end())
        {
            return nullptr;
        }

        return it->second.Get();
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

    void* MapBuffer(int32_t handle, D3D11_MAP mapType)
    {
        //LOG_INFO() << "Mapping: " << handle;
        ID3D11Buffer* stagingBuffer = GetStagingBuffer(handle);
        if (!stagingBuffer)
        {
            return nullptr;
        }

        D3D11_MAPPED_SUBRESOURCE mapped = {};
        if (FAILED(m_context->Map(stagingBuffer, 0u, mapType, 0u, &mapped)))
        {
            return nullptr;
        }

        m_bufferMappedPtrMap[handle] = mapped.pData;
        return mapped.pData;
    }

    void UnmapBuffer(int32_t handle)
    {
        //LOG_INFO() << "UnMapping: " << handle;

        ID3D11Buffer* stagingBuffer = GetStagingBuffer(handle);
        if (!stagingBuffer)
        {
            return;
        }

        auto it = m_bufferMappedPtrMap.find(handle);
        if (it == m_bufferMappedPtrMap.end())
        {
            return;
        }

        m_context->Unmap(stagingBuffer, 0u);
        m_bufferMappedPtrMap.erase(it);
    }

    bool CopyStagingToGPU(int32_t handle)
    {
        ID3D11Buffer* gpuBuffer = GetBuffer(handle);
        ID3D11Buffer* stagingBuffer = GetStagingBuffer(handle);
        if (!gpuBuffer || !stagingBuffer)
        {
            return false;
        }

        m_context->CopyResource(gpuBuffer, stagingBuffer);
        return true;
    }

    bool CopyGPUToStaging(int32_t handle)
    {
        ID3D11Buffer* gpuBuffer = GetBuffer(handle);
        ID3D11Buffer* stagingBuffer = GetStagingBuffer(handle);
        if (!gpuBuffer || !stagingBuffer)
        {
            return false;
        }

        m_context->CopyResource(stagingBuffer, gpuBuffer);
        return true;
    }

    ID3D11UnorderedAccessView* GetBufferUAV(int32_t handle) const
    {
        auto it = m_bufferUAVMap.find(handle);
        if (it == m_bufferUAVMap.end())
        {
            return nullptr;
        }

        return it->second.Get();
    }

    ID3D11ShaderResourceView* GetBufferSRV(int32_t handle) const
    {
        auto it = m_bufferSRVMap.find(handle);
        if (it == m_bufferSRVMap.end())
        {
            return nullptr;
        }

        return it->second.Get();
    }

    ID3D11Buffer* GetCBuffer(int32_t handle) const
    {
        auto it = m_cbufferMap.find(handle);
        if (it == m_cbufferMap.end())
        {
            return nullptr;
        }

        return it->second.Get();
    }

    ID3D11ComputeShader* GetShader(const std::string& name) const
    {
        auto it = m_shaderMap.find(name);
        if (it == m_shaderMap.end())
        {
            return nullptr;
        }

        return it->second.Get();
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

        CBufferParams params = itParams->second;
        std::memset(params.optionalParams, 0, sizeof(params.optionalParams));
        if (optionalParams && optionalCount > 0u)
        {
            const uint32_t count = optionalCount < 16u ? optionalCount : 16u;
            std::memcpy(params.optionalParams, optionalParams, sizeof(uint32_t) * count);
        }

        D3D11_MAPPED_SUBRESOURCE mapped = {};
        HRESULT hr = m_context->Map(it->second.Get(), 0u, D3D11_MAP_WRITE_DISCARD, 0u, &mapped);
        if (FAILED(hr))
        {
            LOG_INFO() << "SetCBufferOptionalParams failed: Map() failed for handle: " << handle;
            return false;
        }

        std::memcpy(mapped.pData, &params, sizeof(params));
        m_context->Unmap(it->second.Get(), 0u);
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
        if (!m_device || !m_context)
        {
            return false;
        }

        D3D11_QUERY_DESC desc = {};
        desc.Query = D3D11_QUERY_EVENT;

        ComPtr<ID3D11Query> query;
        if (FAILED(m_device->CreateQuery(&desc, &query)))
        {
            return false;
        }

        m_context->End(query.Get());
        m_context->Flush();
        while (S_OK != m_context->GetData(query.Get(), nullptr, 0u, 0u))
        {
        }

        return true;
    }

    bool LoadComputeShaderFromFile(std::string _filepath, std::string _name)
    {
		//LOG_INFO() << "Loading compute shader from file: " << _filepath;

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
        // Compile the compute shader from the in-memory HLSL string.
        ComPtr<ID3DBlob> csBlob;
        ComPtr<ID3DBlob> errorBlob;
        UINT compileFlags = D3DCOMPILE_ENABLE_STRICTNESS;
    #if defined(_DEBUG)
        compileFlags |= D3DCOMPILE_DEBUG;
    #endif
        HRESULT hr = D3DCompile(_shader,
                        std::strlen(_shader),
                        nullptr,
                        nullptr,
                        nullptr,
                        "CSMain",
                        "cs_5_0",
                        compileFlags,
                        0u,
                        &csBlob,
                        &errorBlob);
        if (FAILED(hr))
        {
            if (errorBlob)
            {
                const void* errorData = errorBlob->GetBufferPointer();
                const size_t errorSize = errorBlob->GetBufferSize();
				LOG_INFO() << "Compute Shader Compilation Error:";
                LOG_INFO() << _name;
                if (errorData && errorSize > 0u)
                {
                    std::fwrite(errorData, 1, errorSize, stderr);
                    std::fputc('\n', stderr);
                }
            }
			LOG_INFO() << "Failed to compile compute shader: " << _name;
            assert(false);
            return false;
        }

        // Create the compute shader object.
        ComPtr<ID3D11ComputeShader> computeShader;
        hr = m_device->CreateComputeShader(csBlob->GetBufferPointer(),
                                        csBlob->GetBufferSize(),
                                        nullptr,
                                        &computeShader);
        if (FAILED(hr))
        {
            LOG_INFO() << "Failed to create compute shader.";
            assert(false);
            return false;
        }

        errorBlob.Reset();
        csBlob.Reset();

        //LOG_INFO() << "Adding to shader map:" << _name;
        m_shaderMap[_name] = computeShader;
        m_shaders.push_back(computeShader);

        return true;
    }

    void DumpInfoQueueErrors()
    {
        #ifdef _DEBUG
        if (!m_device)
        {
            return;
        }

        ComPtr<ID3D11InfoQueue> infoQueue;
        if (SUCCEEDED(m_device.As(&infoQueue)))
        {
            const UINT64 count = infoQueue->GetNumStoredMessagesAllowedByRetrievalFilter();
            for (UINT64 i = 0; i < count; ++i)
            {
                SIZE_T len = 0;
                infoQueue->GetMessage(i, nullptr, &len);
                if (len == 0)
                {
                    continue;
                }
                std::vector<char> bytes(len);
                auto* msg = reinterpret_cast<D3D11_MESSAGE*>(bytes.data());
                infoQueue->GetMessage(i, msg, &len);
                if (msg && msg->pDescription)
                {
                    LOG_ERROR() << "D3D11: " << msg->pDescription;
                }
            }
            infoQueue->ClearStoredMessages();
        }
        #endif
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

    DirectX11Manager() = default;
    ~DirectX11Manager()
    {
        LifetimeState().alive = false;
        m_init = false;
        m_context = nullptr;
        m_device.Reset();
    }

    public:

    ComPtr<ID3D11Device>        m_device;
    ID3D11DeviceContext*        m_context = nullptr;
    
    std::vector<ComPtr<ID3D11ComputeShader>> m_shaders;
    std::map<std::string, ComPtr<ID3D11ComputeShader>> m_shaderMap;

    int32_t                                                 m_bufferHandles = -1;
    std::vector<ComPtr<ID3D11Buffer>>                       m_buffers;
    std::vector<ComPtr<ID3D11ShaderResourceView>>           m_bufferSRVs;
    std::map<int32_t, ComPtr<ID3D11Buffer>>                 m_bufferMap;
    std::map<int32_t, uint64_t>                             m_bufferSizesMap;
    std::map<int32_t, ComPtr<ID3D11Buffer>>                 m_bufferStagingMap;
    std::map<int32_t, ComPtr<ID3D11ShaderResourceView>>     m_bufferSRVMap;
    std::map<int32_t, ComPtr<ID3D11UnorderedAccessView>>    m_bufferUAVMap;
    std::map<int32_t, void*>                                m_bufferMappedPtrMap;

    int32_t                                  m_cbufferHandles = -1;
    std::map<int32_t, ComPtr<ID3D11Buffer>>  m_cbufferMap;
    std::map<int32_t, CBufferParams>         m_cbufferParamsMap;
    std::vector<int32_t>                     m_cbufferFreeList;

    uint64_t m_allocatedBytes;

    bool m_init = false;
    
};


#endif

