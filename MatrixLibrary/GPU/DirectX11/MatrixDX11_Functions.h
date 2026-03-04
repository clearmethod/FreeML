#pragma once

#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/GPU/DirectX11/MatrixDX11.h>
#include <MatrixLibrary/GPU/DirectX11/DirectX11Manager.h>
#include <cstring>

#ifdef _WIN32

constexpr bool GPUSyncCalls = false;

template<class T>
inline bool SyncCBuffer(MatrixDX11<T>* mat)
{
    if (!mat || !mat->m_gInstance)
    {
        return false;
    }

    DirectX11Manager::CBufferParams params = {};
    if (!mat->m_gInstance->GetCBufferParams(mat->m_dataHandles.m_cbufferHandle, &params))
    {
        return false;
    }

    return mat->m_gInstance->UpdateCachedCBuffer(mat->m_dataHandles.m_cbufferHandle,
                                                 mat->GetDimsX(),
                                                 mat->GetDimsY(),
                                                 mat->GetDimsZ(),
                                                 mat->m_offset,
                                                 params.uniqueId);
}

template<class T>
void Copy(MatrixDX11<T>* _out, MatrixDX11<T>* _in)
{
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    assert(_out->GetElementCount() == _in->GetElementCount());

    DirectX11Manager* manager = _out->m_gInstance;
    if (!SyncCBuffer(_out) || !SyncCBuffer(_in))
    {
        return;
    }


	auto* outPtr = manager->GetMappedPointer(_out->m_dataHandles.m_bufferHandle);
	auto* inPtr  = manager->GetMappedPointer(_in->m_dataHandles.m_bufferHandle);
    if(outPtr && inPtr)
    {
        std::memcpy(outPtr, inPtr, sizeof(T) * _in->GetElementCount());
    }
    else
    {
        _out->Unmap();
        _in->Unmap();

        ID3D11Buffer* outBuffer = manager ? manager->GetBuffer(_out->m_dataHandles.m_bufferHandle) : nullptr;
        ID3D11Buffer* inBuffer  = manager ? manager->GetBuffer(_in->m_dataHandles.m_bufferHandle) : nullptr;
        if (manager && outBuffer && inBuffer)
        {
            manager->GetContext()->CopyResource(outBuffer, inBuffer);
            return;
        }
    }
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if(GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();

}

template<class T>
void CopyRange(MatrixDX11<T>* _out,
               MatrixDX11<T>* _in,
               uint32_t outOffset,
               uint32_t inOffset,
               uint32_t count)
{
    assert(_out);
    assert(_in);
    if (count == 0u)
    {
        return;
    }
    assert(outOffset + count <= _out->GetElementCount());
    assert(inOffset + count <= _in->GetElementCount());

    _out->Unmap();
    _in->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("copy_range") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_in))
    {
        return;
    }

    const uint32_t optionalParams[3] = { outOffset, inOffset, count };
    if (!manager->SetCBufferOptionalParams(_out->m_dataHandles.m_cbufferHandle, optionalParams, 3u))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* inCBuffer = manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !inCBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* inSRV = manager->GetBufferSRV(_in->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!inSRV || !outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { inSRV };
    context->CSSetShaderResources(0u, 1u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { outCBuffer, inCBuffer };
    context->CSSetConstantBuffers(0u, 2u, cbs);

    uint32_t paramsOut[3];
    DirectX11Manager::CBufferParams paramout;
    DirectX11Manager::Instance()->GetCBufferOptionalParams(_out->m_dataHandles.m_cbufferHandle, &paramsOut[0], 3);
    DirectX11Manager::Instance()->GetCBufferParams(_out->m_dataHandles.m_cbufferHandle, &paramout);

    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (count + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
    context->CSSetShaderResources(0u, 1u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();

}

template<class T>
void GatherRows(MatrixDX11<T>* _out, MatrixDX11<T>* _src, MatrixDX11<T>* _indices)
{
    assert(_out);
    assert(_src);
    assert(_indices);
    assert(_indices->GetDimsX() == 1u);
    assert(_out->GetDimsX() == _src->GetDimsX());
    assert(_out->GetDimsY() == _indices->GetDimsY());
    assert(_out->GetDimsZ() == _indices->GetDimsZ());
    assert(_src->GetDimsZ() == 1u || _src->GetDimsZ() == _out->GetDimsZ());

    _out->Unmap();
    _src->Unmap();
    _indices->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("gather_rows") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_src) || !SyncCBuffer(_indices))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* srcCBuffer = manager->GetCBuffer(_src->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* idxCBuffer = manager->GetCBuffer(_indices->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !srcCBuffer || !idxCBuffer)
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* idxSRV = manager->GetBufferSRV(_indices->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* srcSRV = manager->GetBufferSRV(_src->m_dataHandles.m_bufferHandle);
    if (!outUAV || !idxSRV || !srcSRV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11ShaderResourceView* srvs[] = { idxSRV, srcSRV };
    context->CSSetShaderResources(0u, 2u, srvs);
    ID3D11Buffer* cbs[] = { outCBuffer, srcCBuffer, idxCBuffer };
    context->CSSetConstantBuffers(0u, 3u, cbs);

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
    context->CSSetShaderResources(0u, 2u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void ScatterRows(MatrixDX11<T>* _out, MatrixDX11<T>* _src, MatrixDX11<T>* _indices)
{
    assert(_out);
    assert(_src);
    assert(_indices);
    assert(_indices->GetDimsX() == 1u);
    assert(_src->GetDimsX() == _out->GetDimsX());
    assert(_src->GetDimsY() == _indices->GetDimsY());
    assert(_src->GetDimsZ() == _indices->GetDimsZ());
    assert(_out->GetDimsZ() == 1u || _out->GetDimsZ() == _src->GetDimsZ());

    _out->Unmap();
    _src->Unmap();
    _indices->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("scatter_rows") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_src) || !SyncCBuffer(_indices))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* srcCBuffer = manager->GetCBuffer(_src->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* idxCBuffer = manager->GetCBuffer(_indices->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !srcCBuffer || !idxCBuffer)
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* idxSRV = manager->GetBufferSRV(_indices->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* srcSRV = manager->GetBufferSRV(_src->m_dataHandles.m_bufferHandle);
    if (!outUAV || !idxSRV || !srcSRV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11ShaderResourceView* srvs[] = { idxSRV, srcSRV };
    context->CSSetShaderResources(0u, 2u, srvs);
    ID3D11Buffer* cbs[] = { outCBuffer, srcCBuffer, idxCBuffer };
    context->CSSetConstantBuffers(0u, 3u, cbs);

    const uint32_t elementCount = _src->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
    context->CSSetShaderResources(0u, 2u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void ScatterAddRows(MatrixDX11<T>* _out, MatrixDX11<T>* _src, MatrixDX11<T>* _indices)
{
    assert(_out);
    assert(_src);
    assert(_indices);
    assert(_indices->GetDimsX() == 1u);
    assert(_src->GetDimsX() == _out->GetDimsX());
    assert(_src->GetDimsY() == _indices->GetDimsY());
    assert(_src->GetDimsZ() == _indices->GetDimsZ());
    assert(_out->GetDimsZ() == 1u || _out->GetDimsZ() == _src->GetDimsZ());

    const uint32_t srcCount = _src->GetElementCount();
    if (srcCount == 0u)
    {
        return;
    }

    _out->Unmap();
    _src->Unmap();
    _indices->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("scatter_add_row") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_src) || !SyncCBuffer(_indices))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* srcCBuffer = manager->GetCBuffer(_src->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* idxCBuffer = manager->GetCBuffer(_indices->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !srcCBuffer || !idxCBuffer)
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* idxSRV = manager->GetBufferSRV(_indices->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* srcSRV = manager->GetBufferSRV(_src->m_dataHandles.m_bufferHandle);
    if (!outUAV || !idxSRV || !srcSRV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11ShaderResourceView* srvs[] = { idxSRV, srcSRV };
    context->CSSetShaderResources(0u, 2u, srvs);
    ID3D11Buffer* cbs[] = { outCBuffer, srcCBuffer, idxCBuffer };
    context->CSSetConstantBuffers(0u, 3u, cbs);

    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (srcCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
    context->CSSetShaderResources(0u, 2u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void SplitQKV(MatrixDX11<T>* _q,
              MatrixDX11<T>* _k,
              MatrixDX11<T>* _v,
              MatrixDX11<T>* _packed)
{
    assert(_q);
    assert(_k);
    assert(_v);
    assert(_packed);

    const uint32_t qCount = _q->GetElementCount();
    if (qCount == 0u)
    {
        return;
    }
    const uint32_t dC = _q->GetDimsX();
    const uint32_t rows = _q->GetDimsY() * _q->GetDimsZ();
    assert(_k->GetElementCount() == qCount);
    assert(_v->GetElementCount() == qCount);
    assert(_packed->GetElementCount() == qCount * 3u);
    assert(_packed->GetDimsX() == dC * 3u);
    assert((_packed->GetDimsY() * _packed->GetDimsZ()) == rows);

    _q->Unmap();
    _k->Unmap();
    _v->Unmap();
    _packed->Unmap();

    DirectX11Manager* manager = _q->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("split_qkv") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_q) || !SyncCBuffer(_k) || !SyncCBuffer(_v) || !SyncCBuffer(_packed))
    {
        return;
    }

    ID3D11Buffer* qCBuffer = manager->GetCBuffer(_q->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* kCBuffer = manager->GetCBuffer(_k->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* vCBuffer = manager->GetCBuffer(_v->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* packedCBuffer = manager->GetCBuffer(_packed->m_dataHandles.m_cbufferHandle);
    if (!qCBuffer || !kCBuffer || !vCBuffer || !packedCBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* packedSRV = manager->GetBufferSRV(_packed->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* qUAV = manager->GetBufferUAV(_q->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* kUAV = manager->GetBufferUAV(_k->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* vUAV = manager->GetBufferUAV(_v->m_dataHandles.m_bufferHandle);
    if (!packedSRV || !qUAV || !kUAV || !vUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { packedSRV };
    context->CSSetShaderResources(1u, 1u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { qUAV, kUAV, vUAV };
    context->CSSetUnorderedAccessViews(0u, 3u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { qCBuffer, kCBuffer, vCBuffer, packedCBuffer };
    context->CSSetConstantBuffers(0u, 4u, cbs);

    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (qCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr, nullptr, nullptr };
    context->CSSetUnorderedAccessViews(0u, 3u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
    context->CSSetShaderResources(1u, 1u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void MergeQKV(MatrixDX11<T>* _packedQKV,
              MatrixDX11<T>* _qHeads,
              MatrixDX11<T>* _kHeads,
              MatrixDX11<T>* _vHeads)
{
    assert(_packedQKV);
    assert(_qHeads);
    assert(_kHeads);
    assert(_vHeads);

    const uint32_t headDim = _qHeads->GetDimsX();
    const uint32_t tokens = _qHeads->GetDimsY();
    const uint32_t nHead = _qHeads->GetDimsZ();
    const uint32_t dC = headDim * nHead;
    const uint32_t qCount = _qHeads->GetElementCount();
    if (qCount == 0u)
    {
        return;
    }

    assert(_kHeads->GetDimsX() == headDim);
    assert(_kHeads->GetDimsY() == tokens);
    assert(_kHeads->GetDimsZ() == nHead);
    assert(_vHeads->GetDimsX() == headDim);
    assert(_vHeads->GetDimsY() == tokens);
    assert(_vHeads->GetDimsZ() == nHead);
    assert(_kHeads->GetElementCount() == qCount);
    assert(_vHeads->GetElementCount() == qCount);

    assert(_packedQKV->GetDimsX() == 3u * dC);
    assert(_packedQKV->GetDimsY() == tokens);
    assert(_packedQKV->GetDimsZ() == 1u);
    assert(_packedQKV->GetElementCount() == qCount * 3u);

    _packedQKV->Unmap();
    _qHeads->Unmap();
    _kHeads->Unmap();
    _vHeads->Unmap();

    DirectX11Manager* manager = _packedQKV->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("merge_qkv") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_packedQKV) || !SyncCBuffer(_qHeads) || !SyncCBuffer(_kHeads) || !SyncCBuffer(_vHeads))
    {
        return;
    }

    ID3D11Buffer* packedCBuffer = manager->GetCBuffer(_packedQKV->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* qCBuffer = manager->GetCBuffer(_qHeads->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* kCBuffer = manager->GetCBuffer(_kHeads->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* vCBuffer = manager->GetCBuffer(_vHeads->m_dataHandles.m_cbufferHandle);
    if (!packedCBuffer || !qCBuffer || !kCBuffer || !vCBuffer)
    {
        return;
    }

    ID3D11UnorderedAccessView* packedUAV = manager->GetBufferUAV(_packedQKV->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* qSRV = manager->GetBufferSRV(_qHeads->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* kSRV = manager->GetBufferSRV(_kHeads->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* vSRV = manager->GetBufferSRV(_vHeads->m_dataHandles.m_bufferHandle);
    if (!packedUAV || !qSRV || !kSRV || !vSRV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { qSRV, kSRV, vSRV };
    context->CSSetShaderResources(1u, 3u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { packedUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { packedCBuffer, qCBuffer, kCBuffer, vCBuffer };
    context->CSSetConstantBuffers(0u, 4u, cbs);

    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (qCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr, nullptr };
    context->CSSetShaderResources(1u, 3u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void Add(MatrixDX11<T>* _out, MatrixDX11<T>* _L, MatrixDX11<T>* _R)
{
    assert(_out);
    assert(_L);
    assert(_R);
    assert(_out->GetElementCount() == _L->GetElementCount());
    assert(_out->GetElementCount() == _R->GetElementCount());
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    _out->Unmap();
    _L->Unmap();
    _R->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    if (!context)
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* leftBuffer = manager->GetBuffer(_L->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* rightBuffer = manager->GetBuffer(_R->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !leftBuffer || !rightBuffer)
    {
        return;
    }

    const bool aliasOutLeft = (outBuffer == leftBuffer);
    const bool aliasOutRight = (outBuffer == rightBuffer);
    if (aliasOutLeft && aliasOutRight)
    {
        return;
    }

    const char* shaderName = (aliasOutLeft || aliasOutRight) ? "add_inplace" : "add";
    ID3D11ComputeShader* shader = manager->GetShader(shaderName);
    if (!shader)
    {
        return;
    }

    if (aliasOutLeft || aliasOutRight)
    {
        MatrixDX11<T>* in = aliasOutLeft ? _R : _L;
        if (!SyncCBuffer(_out) || !SyncCBuffer(in))
        {
            return;
        }

        ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
        ID3D11Buffer* inCBuffer = manager->GetCBuffer(in->m_dataHandles.m_cbufferHandle);
        if (!outCBuffer || !inCBuffer)
        {
            return;
        }

        ID3D11ShaderResourceView* inSRV = manager->GetBufferSRV(in->m_dataHandles.m_bufferHandle);
        ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
        if (!inSRV || !outUAV)
        {
            return;
        }

        context->CSSetShader(shader, nullptr, 0u);
        ID3D11ShaderResourceView* srvs[] = { inSRV };
        context->CSSetShaderResources(1u, 1u, srvs);
        ID3D11UnorderedAccessView* uavs[] = { outUAV };
        context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
        ID3D11Buffer* cbs[] = { outCBuffer, inCBuffer };
        context->CSSetConstantBuffers(0u, 2u, cbs);

        const uint32_t elementCount = _out->GetElementCount();
        const uint32_t threadsPerGroup = 256u;
        const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
        context->Dispatch(groupCount, 1u, 1u);

        ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
        context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
        context->CSSetShaderResources(1u, 1u, nullSRVs);
        context->CSSetShader(nullptr, nullptr, 0u);
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_L) || !SyncCBuffer(_R))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* leftCBuffer = manager->GetCBuffer(_L->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* rightCBuffer = manager->GetCBuffer(_R->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !leftCBuffer || !rightCBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* leftSRV = manager->GetBufferSRV(_L->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* rightSRV = manager->GetBufferSRV(_R->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!leftSRV || !rightSRV || !outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { leftSRV, rightSRV };
    context->CSSetShaderResources(1u, 2u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { outCBuffer, leftCBuffer, rightCBuffer };
    context->CSSetConstantBuffers(0u, 3u, cbs);

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
    context->CSSetShaderResources(1u, 2u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void BroadcastAdd(MatrixDX11<T>* _out, MatrixDX11<T>* _L, MatrixDX11<T>* _R)
{
    assert(_out);
    assert(_L);
    assert(_R);
    assert(_out->GetDimsX() == _L->GetDimsX());
    assert(_out->GetDimsY() == _L->GetDimsY());
    assert(_out->GetDimsZ() == _L->GetDimsZ());
    assert(_R->GetElementCount() >= _L->GetDimsX());

    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    _out->Unmap();
    _L->Unmap();
    _R->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("broadcast_add") : nullptr;
    if (!manager || !context || !shader)
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* leftBuffer = manager->GetBuffer(_L->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !leftBuffer)
    {
        return;
    }

    if (outBuffer != leftBuffer)
    {
        Copy(_out, _L);
        _out->Unmap();
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_R))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* biasCBuffer = manager->GetCBuffer(_R->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !biasCBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* biasSRV = manager->GetBufferSRV(_R->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!biasSRV || !outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { biasSRV };
    context->CSSetShaderResources(1u, 1u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { outCBuffer, biasCBuffer };
    context->CSSetConstantBuffers(0u, 2u, cbs);

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
    context->CSSetShaderResources(1u, 1u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void LayerNormOp(MatrixDX11<T>* _out,
                 MatrixDX11<T>* _in,
                 MatrixDX11<T>* _gamma,
                 MatrixDX11<T>* _beta = nullptr,
                 MatrixDX11<T>* _xHat = nullptr,
                 T _eps = static_cast<T>(1e-5))
{
    assert(_out);
    assert(_in);
    assert(_gamma);
    assert(_out->GetDimsX() == _in->GetDimsX());
    assert(_out->GetDimsY() == _in->GetDimsY());
    assert(_out->GetDimsZ() == _in->GetDimsZ());
    (void)_eps;

    _out->Unmap();
    _in->Unmap();
    _gamma->Unmap();
    if (_beta)
    {
        _beta->Unmap();
    }
    if (_xHat)
    {
        _xHat->Unmap();
    }

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    if (!manager || !context)
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* inBuffer = manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !inBuffer)
    {
        return;
    }
    if (outBuffer == inBuffer)
    {
        return;
    }

    const bool hasBeta = (_beta != nullptr);
    const bool hasXHat = (_xHat != nullptr);
    const char* shaderName = nullptr;
    if (hasXHat)
    {
        shaderName = hasBeta ? "layernorm" : "layernorm_nobeta";
    }
    else
    {
        shaderName = hasBeta ? "layernorm_no_xhat" : "layernorm_nobeta_no_xhat";
    }
    ID3D11ComputeShader* shader = manager->GetShader(shaderName);
    if (!shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_in) || !SyncCBuffer(_gamma))
    {
        return;
    }
    if (hasBeta && !SyncCBuffer(_beta))
    {
        return;
    }
    if (hasXHat && !SyncCBuffer(_xHat))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* inCBuffer = manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* gammaCBuffer = manager->GetCBuffer(_gamma->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* betaCBuffer = hasBeta ? manager->GetCBuffer(_beta->m_dataHandles.m_cbufferHandle) : nullptr;
    ID3D11Buffer* xHatCBuffer = hasXHat ? manager->GetCBuffer(_xHat->m_dataHandles.m_cbufferHandle) : nullptr;
    if (!outCBuffer || !inCBuffer || !gammaCBuffer)
    {
        return;
    }
    if (hasBeta && !betaCBuffer)
    {
        return;
    }
    if (hasXHat && !xHatCBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* inSRV = manager->GetBufferSRV(_in->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* gammaSRV = manager->GetBufferSRV(_gamma->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* betaSRV = hasBeta ? manager->GetBufferSRV(_beta->m_dataHandles.m_bufferHandle) : nullptr;
    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* xHatUAV = hasXHat ? manager->GetBufferUAV(_xHat->m_dataHandles.m_bufferHandle) : nullptr;
    if (!inSRV || !gammaSRV || !outUAV)
    {
        return;
    }
    if (hasBeta && !betaSRV)
    {
        return;
    }
    if (hasXHat && !xHatUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { inSRV, gammaSRV, betaSRV };
    context->CSSetShaderResources(1u, hasBeta ? 3u : 2u, srvs);

    ID3D11Buffer* cbs[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
    uint32_t cbCount = 0u;
    cbs[cbCount++] = outCBuffer;
    cbs[cbCount++] = inCBuffer;
    cbs[cbCount++] = gammaCBuffer;
    if (hasBeta)
    {
        cbs[cbCount++] = betaCBuffer;
    }
    if (hasXHat)
    {
        cbs[cbCount++] = xHatCBuffer;
    }
    context->CSSetConstantBuffers(0u, cbCount, cbs);

    ID3D11UnorderedAccessView* uavs[] = { outUAV, xHatUAV };
    const uint32_t uavCount = hasXHat ? 2u : 1u;
    context->CSSetUnorderedAccessViews(0u, uavCount, uavs, nullptr);

    // One group per row: each group of 256 threads cooperatively reduces its row.
    const uint32_t rowCount = _out->GetDimsY() * _out->GetDimsZ();
    context->Dispatch(rowCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr, nullptr };
    context->CSSetUnorderedAccessViews(0u, uavCount, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr, nullptr };
    context->CSSetShaderResources(1u, hasBeta ? 3u : 2u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void LayerNormBackwardsOp(MatrixDX11<T>* _dX,
                          MatrixDX11<T>* _dY,
                          MatrixDX11<T>* _in,
                          MatrixDX11<T>* _gamma,
                          MatrixDX11<T>* _dGamma,
                          MatrixDX11<T>* _dBeta = nullptr,
                          MatrixDX11<T>* _xHat = nullptr,
                          T _eps = static_cast<T>(1e-5))
{
    assert(_dX);
    assert(_dY);
    assert(_in);
    assert(_gamma);
    assert(_dGamma);
    assert(_dX->GetDimsX() == _in->GetDimsX());
    assert(_dX->GetDimsY() == _in->GetDimsY());
    assert(_dX->GetDimsZ() == _in->GetDimsZ());
    assert(_dY->GetDimsX() == _in->GetDimsX());
    assert(_dY->GetDimsY() == _in->GetDimsY());
    assert(_dY->GetDimsZ() == _in->GetDimsZ());
    assert(_gamma->GetDimsX() == _in->GetDimsX());
    assert(_dGamma->GetDimsX() == _in->GetDimsX());
    (void)_eps;

    if (!_xHat)
    {
        return;
    }

    _dX->Unmap();
    _dY->Unmap();
    _in->Unmap();
    _gamma->Unmap();
    _dGamma->Unmap();
    _xHat->Unmap();
    if (_dBeta)
    {
        _dBeta->Unmap();
    }

    DirectX11Manager* manager = _dX->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    if (!manager || !context)
    {
        return;
    }

    const bool hasBeta = (_dBeta != nullptr);
    const char* paramsShaderName = hasBeta ? "layernorm_backwards_params" : "layernorm_backwards_params_nobeta";
    ID3D11ComputeShader* paramsShader = manager->GetShader(paramsShaderName);
    ID3D11ComputeShader* dxShader = manager->GetShader("layernorm_backwards_dx");
    if (!paramsShader || !dxShader)
    {
        return;
    }

    if (!SyncCBuffer(_dX) || !SyncCBuffer(_dY) || !SyncCBuffer(_in) || !SyncCBuffer(_gamma)
        || !SyncCBuffer(_dGamma) || !SyncCBuffer(_xHat))
    {
        return;
    }
    if (hasBeta && !SyncCBuffer(_dBeta))
    {
        return;
    }

    ID3D11Buffer* dGammaCBuffer = manager->GetCBuffer(_dGamma->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* dBetaCBuffer = hasBeta ? manager->GetCBuffer(_dBeta->m_dataHandles.m_cbufferHandle) : nullptr;
    ID3D11Buffer* dYCBuffer = manager->GetCBuffer(_dY->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* xHatCBuffer = manager->GetCBuffer(_xHat->m_dataHandles.m_cbufferHandle);
    if (!dGammaCBuffer || !dYCBuffer || !xHatCBuffer)
    {
        return;
    }
    if (hasBeta && !dBetaCBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* dYSRV = manager->GetBufferSRV(_dY->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* xHatSRV = manager->GetBufferSRV(_xHat->m_dataHandles.m_bufferHandle);
    if (!dYSRV || !xHatSRV)
    {
        return;
    }

    ID3D11UnorderedAccessView* dGammaUAV = manager->GetBufferUAV(_dGamma->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* dBetaUAV = hasBeta ? manager->GetBufferUAV(_dBeta->m_dataHandles.m_bufferHandle) : nullptr;
    if (!dGammaUAV || (hasBeta && !dBetaUAV))
    {
        return;
    }

    context->CSSetShader(paramsShader, nullptr, 0u);
    ID3D11ShaderResourceView* srvsParams[] = { dYSRV, xHatSRV };
    context->CSSetShaderResources(1u, 2u, srvsParams);

    ID3D11UnorderedAccessView* uavsParams[] = { dGammaUAV, dBetaUAV };
    const uint32_t paramsUavCount = hasBeta ? 2u : 1u;
    context->CSSetUnorderedAccessViews(0u, paramsUavCount, uavsParams, nullptr);

    ID3D11Buffer* cbsParams[4] = { nullptr, nullptr, nullptr, nullptr };
    uint32_t paramsCbCount = 0u;
    cbsParams[paramsCbCount++] = dGammaCBuffer;
    if (hasBeta)
    {
        cbsParams[paramsCbCount++] = dBetaCBuffer;
    }
    cbsParams[paramsCbCount++] = dYCBuffer;
    cbsParams[paramsCbCount++] = xHatCBuffer;
    context->CSSetConstantBuffers(0u, paramsCbCount, cbsParams);

    const uint32_t cols = _dGamma->GetDimsX();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (cols + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVsParams[] = { nullptr, nullptr };
    context->CSSetUnorderedAccessViews(0u, paramsUavCount, nullUAVsParams, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
    context->CSSetShaderResources(1u, 2u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);

    ID3D11Buffer* dXCBuffer = manager->GetCBuffer(_dX->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* inCBuffer = manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* gammaCBuffer = manager->GetCBuffer(_gamma->m_dataHandles.m_cbufferHandle);
    if (!dXCBuffer || !dYCBuffer || !inCBuffer || !gammaCBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* inSRV = manager->GetBufferSRV(_in->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* gammaSRV = manager->GetBufferSRV(_gamma->m_dataHandles.m_bufferHandle);
    if (!inSRV || !gammaSRV)
    {
        return;
    }

    ID3D11UnorderedAccessView* dXUAV = manager->GetBufferUAV(_dX->m_dataHandles.m_bufferHandle);
    if (!dXUAV)
    {
        return;
    }

    context->CSSetShader(dxShader, nullptr, 0u);
    ID3D11ShaderResourceView* srvsDX[] = { dYSRV, inSRV, gammaSRV };
    context->CSSetShaderResources(1u, 3u, srvsDX);
    ID3D11UnorderedAccessView* uavsDX[] = { dXUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavsDX, nullptr);
    ID3D11Buffer* cbsDX[] = { dXCBuffer, dYCBuffer, inCBuffer, gammaCBuffer };
    context->CSSetConstantBuffers(0u, 4u, cbsDX);

    // One group per row, matching the forward dispatch.
    const uint32_t rowCountDX = _dX->GetDimsY() * _dX->GetDimsZ();
    context->Dispatch(rowCountDX, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVsDX[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVsDX, nullptr);
    ID3D11ShaderResourceView* nullSRVsDX[] = { nullptr, nullptr, nullptr };
    context->CSSetShaderResources(1u, 3u, nullSRVsDX);
    context->CSSetShader(nullptr, nullptr, 0u);

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void ScaleAdd(MatrixDX11<T>* _out, MatrixDX11<T>* _in, MatrixDX11<T>* _scale, MatrixDX11<T>* _add)
{
    assert(_out->GetElementCount() == _in->GetElementCount());
    assert(_out->GetElementCount() == _add->GetElementCount());

    DirectX11Manager::Instance()->DumpInfoQueueErrors();


    _out->Unmap();
    _in->Unmap();
    _scale->Unmap();
    _add->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    const bool aliasOutIn = _out->m_dataHandles.m_bufferHandle == _in->m_dataHandles.m_bufferHandle;
    const char* shaderName = aliasOutIn ? "scaleadd_inplace" : "scaleadd";
    ID3D11ComputeShader* shader = manager ? manager->GetShader(shaderName) : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_scale) || !SyncCBuffer(_add) || (!aliasOutIn && !SyncCBuffer(_in)))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* scaleCBuffer = manager->GetCBuffer(_scale->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* addCBuffer = manager->GetCBuffer(_add->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* inCBuffer = aliasOutIn ? nullptr : manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !scaleCBuffer || !addCBuffer || (!aliasOutIn && !inCBuffer))
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* scaleBuffer = manager->GetBuffer(_scale->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* addBuffer = manager->GetBuffer(_add->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* inBuffer = aliasOutIn ? nullptr : manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !scaleBuffer || !addBuffer || (!aliasOutIn && !inBuffer))
    {
        return;
    }

    ID3D11ShaderResourceView* scaleSRV = manager->GetBufferSRV(_scale->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* addSRV = manager->GetBufferSRV(_add->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* inSRV = aliasOutIn ? nullptr : manager->GetBufferSRV(_in->m_dataHandles.m_bufferHandle);
    if (!scaleSRV || !addSRV || (!aliasOutIn && !inSRV))
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    if (aliasOutIn)
    {
        ID3D11ShaderResourceView* srvs[] = { scaleSRV, addSRV };
        context->CSSetShaderResources(1u, 2u, srvs);
    }
    else
    {
        ID3D11ShaderResourceView* srvs[] = { inSRV, scaleSRV, addSRV };
        context->CSSetShaderResources(1u, 3u, srvs);
    }
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    if (aliasOutIn)
    {
        ID3D11Buffer* cbs[] = { outCBuffer, scaleCBuffer, addCBuffer };
        context->CSSetConstantBuffers(0u, 3u, cbs);
    }
    else
    {
        ID3D11Buffer* cbs[] = { outCBuffer, inCBuffer, scaleCBuffer, addCBuffer };
        context->CSSetConstantBuffers(0u, 4u, cbs);
    }

    const uint32_t elementCount     = _out->GetElementCount();
    const uint32_t threadsPerGroup  = 256u;
    const uint32_t groupCount       = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    if (aliasOutIn)
    {
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
        context->CSSetShaderResources(1u, 2u, nullSRVs);
    }
    else
    {
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr, nullptr };
        context->CSSetShaderResources(1u, 3u, nullSRVs);
    }
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();

}

template<class T>
void PerElementMul(MatrixDX11<T>* _out, MatrixDX11<T>* _in, MatrixDX11<T>* _mask)
{
    assert(_out->GetElementCount() == _in->GetElementCount());
    assert(_out->GetElementCount() == _mask->GetElementCount());

    DirectX11Manager::Instance()->DumpInfoQueueErrors();


    _out->Unmap();
    _in->Unmap();
    _mask->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    const bool aliasOutIn = _out->m_dataHandles.m_bufferHandle == _in->m_dataHandles.m_bufferHandle;
    ID3D11ComputeShader* shader = manager ? manager->GetShader(aliasOutIn ? "per_element_mul_inplace" : "per_element_mul") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_mask) || (!aliasOutIn && !SyncCBuffer(_in)))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* maskCBuffer = manager->GetCBuffer(_mask->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* inCBuffer = aliasOutIn ? nullptr : manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !maskCBuffer || (!aliasOutIn && !inCBuffer))
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* maskBuffer = manager->GetBuffer(_mask->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* inBuffer = aliasOutIn ? nullptr : manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !maskBuffer || (!aliasOutIn && !inBuffer))
    {
        return;
    }

    ID3D11ShaderResourceView* maskSRV = manager->GetBufferSRV(_mask->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* inSRV = aliasOutIn ? nullptr : manager->GetBufferSRV(_in->m_dataHandles.m_bufferHandle);
    if (!maskSRV || (!aliasOutIn && !inSRV))
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    if (aliasOutIn)
    {
        ID3D11ShaderResourceView* srvs[] = { maskSRV };
        context->CSSetShaderResources(1u, 1u, srvs);
    }
    else
    {
        ID3D11ShaderResourceView* srvs[] = { inSRV, maskSRV };
        context->CSSetShaderResources(1u, 2u, srvs);
    }
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    if (aliasOutIn)
    {
        ID3D11Buffer* cbs[] = { outCBuffer, maskCBuffer };
        context->CSSetConstantBuffers(0u, 2u, cbs);
    }
    else
    {
        ID3D11Buffer* cbs[] = { outCBuffer, inCBuffer, maskCBuffer };
        context->CSSetConstantBuffers(0u, 3u, cbs);
    }

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    if (aliasOutIn)
    {
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
        context->CSSetShaderResources(1u, 1u, nullSRVs);
    }
    else
    {
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
        context->CSSetShaderResources(1u, 2u, nullSRVs);
    }
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();

}

template<class T>
void TransposeMat(MatrixDX11<T>* _out, MatrixDX11<T>* _in)
{
    assert(_out);
    assert(_in);
    assert(_out->GetDimsX() == _in->GetDimsY());
    assert(_out->GetDimsY() == _in->GetDimsX());
    assert(_out->GetDimsZ() == _in->GetDimsZ());

    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    _out->Unmap();
    _in->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("transpose") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* inBuffer = manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !inBuffer || outBuffer == inBuffer)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_in))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* inCBuffer = manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !inCBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* inSRV = manager->GetBufferSRV(_in->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!inSRV || !outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { inSRV };
    context->CSSetShaderResources(1u, 1u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { outCBuffer, inCBuffer };
    context->CSSetConstantBuffers(0u, 2u, cbs);

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
    context->CSSetShaderResources(1u, 1u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();

}

template<class T>
void GeluMat(MatrixDX11<T>* _out, MatrixDX11<T>* _in)
{
    assert(_out);
    assert(_in);
    assert(_out->GetElementCount() == _in->GetElementCount());

    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    _out->Unmap();
    _in->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    const bool aliasInOut = _out->m_dataHandles.m_bufferHandle == _in->m_dataHandles.m_bufferHandle;
    const char* shaderName = aliasInOut ? "gelu_inplace" : "gelu";
    ID3D11ComputeShader* shader = manager ? manager->GetShader(shaderName) : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || (!aliasInOut && !SyncCBuffer(_in)))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* inCBuffer = aliasInOut ? nullptr : manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || (!aliasInOut && !inCBuffer))
    {
        return;
    }

    ID3D11ShaderResourceView* inSRV = aliasInOut ? nullptr : manager->GetBufferSRV(_in->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if ((!aliasInOut && !inSRV) || !outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    if (aliasInOut)
    {
        ID3D11Buffer* cbs[] = { outCBuffer };
        context->CSSetConstantBuffers(0u, 1u, cbs);
    }
    else
    {
        ID3D11ShaderResourceView* srvs[] = { inSRV };
        context->CSSetShaderResources(1u, 1u, srvs);
        ID3D11Buffer* cbs[] = { outCBuffer, inCBuffer };
        context->CSSetConstantBuffers(0u, 2u, cbs);
    }
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    if (aliasInOut)
    {
        // no SRVs bound
    }
    else
    {
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
        context->CSSetShaderResources(1u, 1u, nullSRVs);
    }
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void GeluDerivtiveMat(MatrixDX11<T>* _out, MatrixDX11<T>* _in)
{
    assert(_out);
    assert(_in);
    assert(_out->GetElementCount() == _in->GetElementCount());
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    _out->Unmap();
    _in->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    const bool aliasInOut = _out->m_dataHandles.m_bufferHandle == _in->m_dataHandles.m_bufferHandle;
    const char* shaderName = aliasInOut ? "gelu_derivative_inplace" : "gelu_derivative";
    ID3D11ComputeShader* shader = manager ? manager->GetShader(shaderName) : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || (!aliasInOut && !SyncCBuffer(_in)))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* inCBuffer = aliasInOut ? nullptr : manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || (!aliasInOut && !inCBuffer))
    {
        return;
    }

    ID3D11ShaderResourceView* inSRV = aliasInOut ? nullptr : manager->GetBufferSRV(_in->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if ((!aliasInOut && !inSRV) || !outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    if (aliasInOut)
    {
        ID3D11Buffer* cbs[] = { outCBuffer };
        context->CSSetConstantBuffers(0u, 1u, cbs);
    }
    else
    {
        ID3D11ShaderResourceView* srvs[] = { inSRV };
        context->CSSetShaderResources(1u, 1u, srvs);
        ID3D11Buffer* cbs[] = { outCBuffer, inCBuffer };
        context->CSSetConstantBuffers(0u, 2u, cbs);
    }
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    if (aliasInOut)
    {
        // no SRVs bound
    }
    else
    {
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
        context->CSSetShaderResources(1u, 1u, nullSRVs);
    }
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();

}

template<class T>
void Scale(MatrixDX11<T>* _out, MatrixDX11<T>* _in, MatrixDX11<T>* _scale)
{
    assert(_out->GetElementCount() == _in->GetElementCount());
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    _out->Unmap();
    _in->Unmap();
    _scale->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    const bool aliasInOut = _out->m_dataHandles.m_bufferHandle == _in->m_dataHandles.m_bufferHandle;
    const char* shaderName = aliasInOut ? "scale_inplace" : "scale";
    ID3D11ComputeShader* shader = manager ? manager->GetShader(shaderName) : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_scale) || (!aliasInOut && !SyncCBuffer(_in)))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* scaleCBuffer = manager->GetCBuffer(_scale->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* inCBuffer = aliasInOut ? nullptr : manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !scaleCBuffer || (!aliasInOut && !inCBuffer))
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* scaleBuffer = manager->GetBuffer(_scale->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* inBuffer = aliasInOut ? nullptr : manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !scaleBuffer || (!aliasInOut && !inBuffer))
    {
        return;
    }

    ID3D11ShaderResourceView* scaleSRV = manager->GetBufferSRV(_scale->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* inSRV = aliasInOut ? nullptr : manager->GetBufferSRV(_in->m_dataHandles.m_bufferHandle);
    if (!scaleSRV || (!aliasInOut && !inSRV))
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    if (aliasInOut)
    {
        ID3D11ShaderResourceView* srvs[] = { scaleSRV };
        context->CSSetShaderResources(1u, 1u, srvs);
    }
    else
    {
        ID3D11ShaderResourceView* srvs[] = { inSRV, scaleSRV };
        context->CSSetShaderResources(1u, 2u, srvs);
    }
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    if (aliasInOut)
    {
        ID3D11Buffer* cbs[] = { outCBuffer, scaleCBuffer };
        context->CSSetConstantBuffers(0u, 2u, cbs);
    }
    else
    {
        ID3D11Buffer* cbs[] = { outCBuffer, inCBuffer, scaleCBuffer };
        context->CSSetConstantBuffers(0u, 3u, cbs);
    }

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    if (aliasInOut)
    {
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
        context->CSSetShaderResources(1u, 1u, nullSRVs);
    }
    else
    {
        ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
        context->CSSetShaderResources(1u, 2u, nullSRVs);
    }
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void Fill(MatrixDX11<T>* _out, MatrixDX11<T>* _val)
{
    assert(_val->GetElementCount() >= 1u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    _out->Unmap();
    _val->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("fill") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_val))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* valCBuffer = manager->GetCBuffer(_val->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !valCBuffer)
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* valBuffer = manager->GetBuffer(_val->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !valBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* valSRV = manager->GetBufferSRV(_val->m_dataHandles.m_bufferHandle);
    if (!valSRV)
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { valSRV };
    context->CSSetShaderResources(1u, 1u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { outCBuffer, valCBuffer };
    context->CSSetConstantBuffers(0u, 2u, cbs);

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
    context->CSSetShaderResources(1u, 1u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void Clear(MatrixDX11<T>* _out)
{
    assert(_out);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    _out->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* deviceContext = manager ? manager->GetContext() : nullptr;
    if (!deviceContext)
    {
        return;
    }

    ID3D11UnorderedAccessView* uav = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!uav)
    {
        return;
    }

    UINT values[4] = { 0u, 0u, 0u, 0u };
    deviceContext->ClearUnorderedAccessViewUint(uav, values);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void Softmax(MatrixDX11<T>* _out, MatrixDX11<T>* _in)
{
    assert(_out->GetElementCount() == _in->GetElementCount());
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    _out->Unmap();
    _in->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("softmax") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_in))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* inCBuffer = manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !inCBuffer)
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* inBuffer = manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !inBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* inSRV = manager->GetBufferSRV(_in->m_dataHandles.m_bufferHandle);
    if (!inSRV)
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { inSRV };
    context->CSSetShaderResources(1u, 1u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { outCBuffer, inCBuffer };
    context->CSSetConstantBuffers(0u, 2u, cbs);

    const uint32_t rowCount = _out->GetDimsY() * _out->GetDimsZ();
    context->Dispatch(rowCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
    context->CSSetShaderResources(1u, 1u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void SoftmaxBackwards(MatrixDX11<T>* _out, MatrixDX11<T>* _softmaxOut, MatrixDX11<T>* _gradIn)
{
    assert(_out->GetElementCount() == _softmaxOut->GetElementCount());
    assert(_out->GetElementCount() == _gradIn->GetElementCount());
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    _out->Unmap();
    _softmaxOut->Unmap();
    _gradIn->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("softmax_backwards") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_softmaxOut) || !SyncCBuffer(_gradIn))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* softCBuffer = manager->GetCBuffer(_softmaxOut->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* gradCBuffer = manager->GetCBuffer(_gradIn->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !softCBuffer || !gradCBuffer)
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* softBuffer = manager->GetBuffer(_softmaxOut->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* gradBuffer = manager->GetBuffer(_gradIn->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !softBuffer || !gradBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* softSRV = manager->GetBufferSRV(_softmaxOut->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* gradSRV = manager->GetBufferSRV(_gradIn->m_dataHandles.m_bufferHandle);
    if (!softSRV || !gradSRV)
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { softSRV, gradSRV };
    context->CSSetShaderResources(1u, 2u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { outCBuffer, softCBuffer, gradCBuffer };
    context->CSSetConstantBuffers(0u, 3u, cbs);

    const uint32_t rowCount = _out->GetDimsY() * _out->GetDimsZ();
    context->Dispatch(rowCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
    context->CSSetShaderResources(1u, 2u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<TransposeMode Mode = TransposeMode::None, class T>
void Mul(MatrixDX11<T>* _out, MatrixDX11<T>* _L, MatrixDX11<T>* _R,
                      uint32_t blockSize = 32, uint32_t numThreads = 12, ThreadPool* pool = nullptr)
{
    DirectX11Manager::Instance()->DumpInfoQueueErrors();
    _out->Unmap();
    _L->Unmap();
    _R->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager->GetContext();
    const char* shaderName = "mul";
    if constexpr (Mode == TransposeMode::Left)
    {
        shaderName = "mul_transposeleft";
    }
    else if constexpr (Mode == TransposeMode::Right)
    {
        shaderName = "mul_transposeright";
    }
    else if constexpr (Mode == TransposeMode::Both)
    {
        shaderName = "mul_transposeboth";
    }
    else if constexpr (Mode == TransposeMode::None)
    {
        shaderName = "mul";
    }
    ID3D11ComputeShader* shader = shaderName ? manager->GetShader(shaderName) : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_L) || !SyncCBuffer(_R))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* leftCBuffer = manager->GetCBuffer(_L->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* rightCBuffer = manager->GetCBuffer(_R->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !leftCBuffer || !rightCBuffer)
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* leftBuffer = manager->GetBuffer(_L->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* rightBuffer = manager->GetBuffer(_R->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !leftBuffer || !rightBuffer)
    {
        return;
    }

    D3D11_BUFFER_DESC leftDesc = {};
    D3D11_BUFFER_DESC rightDesc = {};
    leftBuffer->GetDesc(&leftDesc);
    rightBuffer->GetDesc(&rightDesc);
    uint32_t leftCount = leftDesc.StructureByteStride ? (leftDesc.ByteWidth / leftDesc.StructureByteStride) : 0u;
    uint32_t rightCount = rightDesc.StructureByteStride ? (rightDesc.ByteWidth / rightDesc.StructureByteStride) : 0u;
    if (leftCount == 0u || rightCount == 0u)
    {
        return;
    }

    ID3D11ShaderResourceView* leftSRV = manager->GetBufferSRV(_L->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* rightSRV = manager->GetBufferSRV(_R->m_dataHandles.m_bufferHandle);
    if (!leftSRV || !rightSRV)
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { leftSRV, rightSRV };
    context->CSSetShaderResources(1u, 2u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { outCBuffer, leftCBuffer, rightCBuffer };
    context->CSSetConstantBuffers(0u, 3u, cbs);

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
    context->CSSetShaderResources(1u, 2u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<TransposeMode Mode = TransposeMode::None, class T>
void MatMul_Strided(MatrixDX11<T>* _out,
                    MatrixDX11<T>* _L,
                    MatrixDX11<T>* _R,
                    uint32_t L_rowStride,
                    uint32_t L_colStride,
                    uint32_t R_rowStride,
                    uint32_t R_colStride,
                    uint32_t outRowStride,
                    uint32_t outColStride,
                    uint32_t blockSize = 32,
                    uint32_t numThreads = 12,
                    ThreadPool* pool = nullptr)
{
    DirectX11Manager::Instance()->DumpInfoQueueErrors();
    _out->Unmap();
    _L->Unmap();
    _R->Unmap();

    DirectX11Manager* manager = _out->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    const char* shaderName = "mul_strided";
    if constexpr (Mode == TransposeMode::Left)
    {
        shaderName = "mul_strided_transposeleft";
    }
    else if constexpr (Mode == TransposeMode::Right)
    {
        shaderName = "mul_strided_transposeright";
    }
    else if constexpr (Mode == TransposeMode::Both)
    {
        shaderName = "mul_strided_transposeboth";
    }
    else if constexpr (Mode == TransposeMode::None)
    {
        shaderName = "mul_strided";
    }

    ID3D11ComputeShader* shader = shaderName ? manager->GetShader(shaderName) : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(_out) || !SyncCBuffer(_L) || !SyncCBuffer(_R))
    {
        return;
    }

    uint32_t outParams[2] = { outRowStride, outColStride };
    uint32_t leftParams[2] = { L_rowStride, L_colStride };
    uint32_t rightParams[2] = { R_rowStride, R_colStride };
    if (!manager->SetCBufferOptionalParams(_out->m_dataHandles.m_cbufferHandle, outParams, 2u))
    {
        return;
    }
    if (!manager->SetCBufferOptionalParams(_L->m_dataHandles.m_cbufferHandle, leftParams, 2u))
    {
        return;
    }
    if (!manager->SetCBufferOptionalParams(_R->m_dataHandles.m_cbufferHandle, rightParams, 2u))
    {
        return;
    }

    ID3D11Buffer* outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* leftCBuffer = manager->GetCBuffer(_L->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* rightCBuffer = manager->GetCBuffer(_R->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !leftCBuffer || !rightCBuffer)
    {
        return;
    }

    ID3D11Buffer* outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* leftBuffer = manager->GetBuffer(_L->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* rightBuffer = manager->GetBuffer(_R->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !leftBuffer || !rightBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* leftSRV = manager->GetBufferSRV(_L->m_dataHandles.m_bufferHandle);
    ID3D11ShaderResourceView* rightSRV = manager->GetBufferSRV(_R->m_dataHandles.m_bufferHandle);
    if (!leftSRV || !rightSRV)
    {
        return;
    }

    ID3D11UnorderedAccessView* outUAV = manager->GetBufferUAV(_out->m_dataHandles.m_bufferHandle);
    if (!outUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { leftSRV, rightSRV };
    context->CSSetShaderResources(1u, 2u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { outUAV };
    context->CSSetUnorderedAccessViews(0u, 1u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { outCBuffer, leftCBuffer, rightCBuffer };
    context->CSSetConstantBuffers(0u, 3u, cbs);

    const uint32_t elementCount = _out->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
    context->CSSetUnorderedAccessViews(0u, 1u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr, nullptr };
    context->CSSetShaderResources(1u, 2u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
static inline uint32_t PackFloat(T value)
{
    static_assert(sizeof(T) == sizeof(uint32_t), "PackFloat expects 32-bit float.");
    uint32_t packed = 0u;
    std::memcpy(&packed, &value, sizeof(uint32_t));
    return packed;
}

template<class T>
void AdamWUpdate(MatrixDX11<T>* param,
                 MatrixDX11<T>* grad,
                 MatrixDX11<T>* mt,
                 MatrixDX11<T>* vt,
                 T lr,
                 T beta1,
                 T beta2,
                 T beta1_pow_t,
                 T beta2_pow_t,
                 T eps,
                 T weightDecay)
{
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (!param || !grad || !mt || !vt)
    {
        return;
    }

    param->Unmap();
    grad->Unmap();
    mt->Unmap();
    vt->Unmap();

    if (param->GetElementCount() != grad->GetElementCount()
        || param->GetElementCount() != mt->GetElementCount()
        || param->GetElementCount() != vt->GetElementCount())
    {
        return;
    }

    DirectX11Manager* manager = param->m_gInstance;
    ID3D11DeviceContext* context = manager ? manager->GetContext() : nullptr;
    ID3D11ComputeShader* shader = manager ? manager->GetShader("adam_update") : nullptr;
    if (!context || !shader)
    {
        return;
    }

    if (!SyncCBuffer(param) || !SyncCBuffer(grad) || !SyncCBuffer(mt) || !SyncCBuffer(vt))
    {
        return;
    }

    const uint32_t optionalParams[7] =
    {
        PackFloat(lr),
        PackFloat(beta1),
        PackFloat(beta2),
        PackFloat(beta1_pow_t),
        PackFloat(beta2_pow_t),
        PackFloat(eps),
        PackFloat(weightDecay)
    };
    if (!manager->SetCBufferOptionalParams(param->m_dataHandles.m_cbufferHandle, optionalParams, 7u))
    {
        return;
    }

    ID3D11Buffer* paramCBuffer = manager->GetCBuffer(param->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* gradCBuffer = manager->GetCBuffer(grad->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* mtCBuffer = manager->GetCBuffer(mt->m_dataHandles.m_cbufferHandle);
    ID3D11Buffer* vtCBuffer = manager->GetCBuffer(vt->m_dataHandles.m_cbufferHandle);
    if (!paramCBuffer || !gradCBuffer || !mtCBuffer || !vtCBuffer)
    {
        return;
    }

    ID3D11Buffer* paramBuffer = manager->GetBuffer(param->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* gradBuffer = manager->GetBuffer(grad->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* mtBuffer = manager->GetBuffer(mt->m_dataHandles.m_bufferHandle);
    ID3D11Buffer* vtBuffer = manager->GetBuffer(vt->m_dataHandles.m_bufferHandle);
    if (!paramBuffer || !gradBuffer || !mtBuffer || !vtBuffer)
    {
        return;
    }

    if (paramBuffer == gradBuffer || paramBuffer == mtBuffer || paramBuffer == vtBuffer
        || gradBuffer == mtBuffer || gradBuffer == vtBuffer || mtBuffer == vtBuffer)
    {
        return;
    }

    ID3D11ShaderResourceView* gradSRV = manager->GetBufferSRV(grad->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* paramUAV = manager->GetBufferUAV(param->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* mtUAV = manager->GetBufferUAV(mt->m_dataHandles.m_bufferHandle);
    ID3D11UnorderedAccessView* vtUAV = manager->GetBufferUAV(vt->m_dataHandles.m_bufferHandle);
    if (!gradSRV || !paramUAV || !mtUAV || !vtUAV)
    {
        return;
    }

    context->CSSetShader(shader, nullptr, 0u);
    ID3D11ShaderResourceView* srvs[] = { gradSRV };
    context->CSSetShaderResources(1u, 1u, srvs);
    ID3D11UnorderedAccessView* uavs[] = { paramUAV, mtUAV, vtUAV };
    context->CSSetUnorderedAccessViews(0u, 3u, uavs, nullptr);
    ID3D11Buffer* cbs[] = { paramCBuffer, gradCBuffer, mtCBuffer, vtCBuffer };
    context->CSSetConstantBuffers(0u, 4u, cbs);

    const uint32_t elementCount = param->GetElementCount();
    const uint32_t threadsPerGroup = 256u;
    const uint32_t groupCount = (elementCount + threadsPerGroup - 1u) / threadsPerGroup;
    context->Dispatch(groupCount, 1u, 1u);

    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr, nullptr, nullptr };
    context->CSSetUnorderedAccessViews(0u, 3u, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRVs[] = { nullptr };
    context->CSSetShaderResources(1u, 1u, nullSRVs);
    context->CSSetShader(nullptr, nullptr, 0u);
    DirectX11Manager::Instance()->DumpInfoQueueErrors();

    if (GPUSyncCalls)
        DirectX11Manager::Instance()->WaitForGPU();
}

template<class T>
void AdamUpdate(MatrixDX11<T>* param,
                MatrixDX11<T>* grad,
                MatrixDX11<T>* mt,
                MatrixDX11<T>* vt,
                T lr,
                T beta1,
                T beta2,
                T beta1_pow_t,
                T beta2_pow_t,
                T eps)
{
    AdamWUpdate(param,
                grad,
                mt,
                vt,
                lr,
                beta1,
                beta2,
                beta1_pow_t,
                beta2_pow_t,
                eps,
                T(0));
}


#endif
