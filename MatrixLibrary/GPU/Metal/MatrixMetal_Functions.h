#pragma once

#ifdef __APPLE__

#include <MatrixLibrary/GPU/Metal/MatrixMetal.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/MatrixManager.h>
#include <cstring>

// Copy constants into the command stream: later updates or cache reuse must not
// change parameters of dispatches that have already been encoded.
inline void MetalBindCBuffer(Obj encoder, Obj cbuffer, uint64_t index)
{
    const void* params = Send<const void*>(cbuffer, "contents");
    assert(params);
    Send<void>(encoder, "setBytes:length:atIndex:", params,
               sizeof(MetalManager::CBufferParams), index);
}

// The larger cooperative tile pays off on substantial products. Keep the
// scalar tile for small matrices; benchmark crossover varies with the GPU.
inline Obj MetalMatMulSIMDPipeline(MetalManager* manager, uint32_t rows,
                                  uint32_t columns, uint32_t inner)
{
    if (uint64_t(rows) * columns * inner < 1024u * 1024u)
        return nullptr;
    Obj pipeline = manager->GetPipeline("mul_simdgroup");
    return pipeline && Send<uint64_t>(pipeline, "threadExecutionWidth") == 32u
        ? pipeline : nullptr;
}

template<class T>
inline bool MetalSyncCBuffer(MatrixMetal<T>* mat)
{
    if (!mat || !mat->m_gInstance)
    {
        return false;
    }

    MetalManager::CBufferParams params = {};
    if (!mat->m_gInstance->GetCBufferParams(mat->m_dataHandles.m_cbufferHandle, &params))
    {
        return false;
    }

    return mat->m_gInstance->UpdateCachedCBuffer(mat->m_dataHandles.m_cbufferHandle,
                                                 mat->GetDimsX(),
                                                 mat->GetDimsY(),
                                                 mat->GetDimsZ(),
                                                 mat->GetOffset(),
                                                 params.uniqueId);
}

template<class T>
void GatherRows(MatrixMetal<T>* out, MatrixMetal<T>* src, MatrixMetal<T>* indices)
{
    assert(out && src && indices);
    assert(indices->GetDimsX() == 1u);
    assert(out->GetDimsX() == src->GetDimsX());
    assert(indices->GetDimsY() == out->GetDimsY());
    assert(indices->GetDimsZ() == out->GetDimsZ());
    assert(src->GetDimsZ() == 1u || src->GetDimsZ() == out->GetDimsZ());
    const uint32_t count = out->GetElementCount();
    if (count == 0u) return;
    auto* manager = out->m_gInstance;
    out->Unmap(); src->Unmap(); indices->Unmap();
    if (!MetalSyncCBuffer(out) || !MetalSyncCBuffer(src) || !MetalSyncCBuffer(indices))
        return;
    Obj pipeline = manager->GetPipeline("gather_rows");
    Obj encoder = manager->GetEncoder();
    Obj outputBuffer = manager->GetBuffer(out->m_dataHandles.m_bufferHandle);
    Obj sourceBuffer = manager->GetBuffer(src->m_dataHandles.m_bufferHandle);
    Obj indexBuffer = manager->GetBuffer(indices->m_dataHandles.m_bufferHandle);
    if (!pipeline || !encoder || !outputBuffer || !sourceBuffer || !indexBuffer)
        return;
    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outputBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", sourceBuffer, 0ull, 1ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", indexBuffer, 0ull, 2ull);
    MetalBindCBuffer(encoder, manager->GetCBuffer(out->m_dataHandles.m_cbufferHandle), 3ull);
    MetalBindCBuffer(encoder, manager->GetCBuffer(src->m_dataHandles.m_cbufferHandle), 4ull);
    MetalBindCBuffer(encoder, manager->GetCBuffer(indices->m_dataHandles.m_cbufferHandle), 5ull);
    MTLSize threads = {count, 1u, 1u}, group = {256u, 1u, 1u};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, group);
    manager->WorkSubmitted();
}

// Each thread owns one feature in one destination batch, so repeated indices
// accumulate in source order without atomics or CPU staging transfers.
template<class T>
void ScatterAddRows(MatrixMetal<T>* out, MatrixMetal<T>* src, MatrixMetal<T>* indices)
{
    assert(out && src && indices);
    assert(indices->GetDimsX() == 1u);
    assert(src->GetDimsX() == out->GetDimsX());
    assert(src->GetDimsY() == indices->GetDimsY());
    assert(src->GetDimsZ() == indices->GetDimsZ());
    assert(out->GetDimsZ() == 1u || out->GetDimsZ() == src->GetDimsZ());
    auto* manager = out->m_gInstance;
    out->Unmap(); src->Unmap(); indices->Unmap();
    if (!MetalSyncCBuffer(out) || !MetalSyncCBuffer(src) || !MetalSyncCBuffer(indices))
        return;
    Obj pipeline = manager->GetPipeline("scatter_add_rows");
    Obj encoder = manager->GetEncoder();
    Obj outputBuffer = manager->GetBuffer(out->m_dataHandles.m_bufferHandle);
    Obj sourceBuffer = manager->GetBuffer(src->m_dataHandles.m_bufferHandle);
    Obj indexBuffer = manager->GetBuffer(indices->m_dataHandles.m_bufferHandle);
    if (!pipeline || !encoder || !outputBuffer || !sourceBuffer || !indexBuffer)
        return;
    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outputBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", sourceBuffer, 0ull, 1ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", indexBuffer, 0ull, 2ull);
    MetalBindCBuffer(encoder, manager->GetCBuffer(out->m_dataHandles.m_cbufferHandle), 3ull);
    MetalBindCBuffer(encoder, manager->GetCBuffer(src->m_dataHandles.m_cbufferHandle), 4ull);
    MetalBindCBuffer(encoder, manager->GetCBuffer(indices->m_dataHandles.m_cbufferHandle), 5ull);
    const uint32_t count = out->GetDimsX() * out->GetDimsZ();
    if (count == 0u)
        return;
    MTLSize threads = {count, 1u, 1u}, group = {256u, 1u, 1u};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, group);
    manager->WorkSubmitted();
}

template<class T>
void Copy(MatrixMetal<T>* _out, MatrixMetal<T>* _in)
{
    assert(_out);
    assert(_in);
    assert(_out->GetElementCount() == _in->GetElementCount());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    if (!MetalSyncCBuffer(_out) || !MetalSyncCBuffer(_in))
    {
        return;
    }

    auto* outPtr = manager->GetMappedPointer(_out->m_dataHandles.m_bufferHandle);
    auto* inPtr = manager->GetMappedPointer(_in->m_dataHandles.m_bufferHandle);
    if (outPtr && inPtr)
    {
        manager->MarkMappingDirty(_out->m_dataHandles.m_bufferHandle);
        std::memcpy(static_cast<T*>(outPtr) + _out->GetOffset(),
                    static_cast<const T*>(inPtr) + _in->GetOffset(),
                    sizeof(T) * _in->GetElementCount());
    }
    else
    {
        _out->Unmap();
        _in->Unmap();
        manager->CopyBuffer(_out->m_dataHandles.m_bufferHandle, _in->m_dataHandles.m_bufferHandle,
                            uint64_t(_out->GetOffset()) * sizeof(T),
                            uint64_t(_in->GetOffset()) * sizeof(T),
                            uint64_t(_out->GetElementCount()) * sizeof(T));
    }

    manager->DumpInfoQueueErrors();
}


template<class T>
void Clear(MatrixMetal<T>* _out)
{
    assert(_out);

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    manager->ClearBuffer(_out->m_dataHandles.m_bufferHandle,
                         _out->GetOffset(), _out->GetElementCount());
    manager->DumpInfoQueueErrors();
}


template<class T>
void SoftmaxBackwards(MatrixMetal<T>* _out, MatrixMetal<T>* _softmaxOut, MatrixMetal<T>* _gradIn)
{
    assert(_out);
    assert(_softmaxOut);
    assert(_gradIn);
    assert(_out->GetElementCount() == _softmaxOut->GetElementCount());
    assert(_out->GetElementCount() == _gradIn->GetElementCount());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _softmaxOut->Unmap();
    _gradIn->Unmap();

    Obj pipeline = manager->GetPipeline("softmax_backwards");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!MetalSyncCBuffer(_out) || !MetalSyncCBuffer(_softmaxOut) || !MetalSyncCBuffer(_gradIn))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj softBuffer = manager->GetBuffer(_softmaxOut->m_dataHandles.m_bufferHandle);
    Obj gradBuffer = manager->GetBuffer(_gradIn->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj softCBuffer = manager->GetCBuffer(_softmaxOut->m_dataHandles.m_cbufferHandle);
    Obj gradCBuffer = manager->GetCBuffer(_gradIn->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !softBuffer || !gradBuffer || !outCBuffer || !softCBuffer || !gradCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", softBuffer, 0ull, 1ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", gradBuffer, 0ull, 2ull);
    MetalBindCBuffer(encoder, outCBuffer, 3ull);
    MetalBindCBuffer(encoder, softCBuffer, 4ull);
    MetalBindCBuffer(encoder, gradCBuffer, 5ull);

    constexpr uint64_t groupSize = 256u;
    Send<void>(encoder, "setThreadgroupMemoryLength:atIndex:", groupSize * sizeof(float), 0ull);

    const uint32_t rowCount = _out->GetDimsY() * _out->GetDimsZ();
    MTLSize threads = {rowCount * groupSize, 1, 1};
    MTLSize threadsPerThreadgroup = {groupSize, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
void AttentionScoresBackwards(MatrixMetal<T>* out, MatrixMetal<T>* probs,
                              MatrixMetal<T>* grad, MatrixMetal<T>* scale)
{
    assert(out && probs && grad && scale);
    assert(out->GetDimsX() == out->GetDimsY());
    assert(out->GetDimsX() == probs->GetDimsX() && out->GetDimsY() == probs->GetDimsY()
           && out->GetDimsZ() == probs->GetDimsZ());
    assert(out->GetDimsX() == grad->GetDimsX() && out->GetDimsY() == grad->GetDimsY()
           && out->GetDimsZ() == grad->GetDimsZ());
    assert(scale->GetElementCount() == 1u);
    if (out->GetElementCount() == 0u) return;
    auto* manager = out->m_gInstance;
    out->Unmap(); probs->Unmap(); grad->Unmap(); scale->Unmap();
    if (!MetalSyncCBuffer(out) || !MetalSyncCBuffer(probs)
        || !MetalSyncCBuffer(grad) || !MetalSyncCBuffer(scale)) return;
    Obj pipeline = manager->GetPipeline("attention_scores_backwards");
    Obj encoder = manager->GetEncoder();
    if (!pipeline || !encoder) return;
    Send<void>(encoder, "setComputePipelineState:", pipeline);
    MatrixMetal<T>* matrices[] = {out, probs, grad, scale};
    for (uint64_t i = 0; i < 4; ++i)
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:",
                   manager->GetBuffer(matrices[i]->m_dataHandles.m_bufferHandle), 0ull, i);
        MetalBindCBuffer(encoder, manager->GetCBuffer(matrices[i]->m_dataHandles.m_cbufferHandle), i + 4ull);
    }
    constexpr uint64_t groupSize = 256u;
    Send<void>(encoder, "setThreadgroupMemoryLength:atIndex:", groupSize * sizeof(float), 0ull);
    MTLSize groups = {out->GetDimsY() * out->GetDimsZ(), 1, 1};
    MTLSize threads = {groupSize, 1, 1};
    Send<void>(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", groups, threads);
    manager->WorkSubmitted();
}

template<class T>
void SumColumns(MatrixMetal<T>* _out, MatrixMetal<T>* _in)
{
    assert(_out);
    assert(_in);
    assert(_out->GetDimsX() == _in->GetDimsX());
    assert(_out->GetDimsY() > 0);

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _in->Unmap();

    Obj parallelPipeline = _in->GetDimsY() >= 64u
        ? manager->GetPipeline("sum_columns_parallel") : nullptr;
    const bool parallel = parallelPipeline != nullptr;
    Obj pipeline = parallel ? parallelPipeline : manager->GetPipeline("sum_columns");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!MetalSyncCBuffer(_out) || !MetalSyncCBuffer(_in))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj inBuffer = manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj inCBuffer = manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !inBuffer || !outCBuffer || !inCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 1ull);
    MetalBindCBuffer(encoder, outCBuffer, 2ull);
    MetalBindCBuffer(encoder, inCBuffer, 3ull);

    MTLSize threads = {_out->GetDimsX(), 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    if (parallel)
        Send<void>(encoder, "dispatchThreadgroups:threadsPerThreadgroup:",
            MTLSize{(_out->GetDimsX()+31u)/32u,1u,1u},MTLSize{32u,8u,1u});
    else
        Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}


template<class T>
void Softmax(MatrixMetal<T>* _out, MatrixMetal<T>* _in)
{
    assert(_out);
    assert(_in);
    assert(_out->GetElementCount() == _in->GetElementCount());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _in->Unmap();

    Obj pipeline = manager->GetPipeline("softmax");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID()))
        || !manager->UpdateCachedCBuffer(_in->m_dataHandles.m_cbufferHandle,
                                         _in->GetDimsX(),
                                         _in->GetDimsY(),
                                         _in->GetDimsZ(),
                                         _in->GetOffset(),
                                         static_cast<uint32_t>(_in->GetID())))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj inBuffer = manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj inCBuffer = manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !inBuffer || !outCBuffer || !inCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 1ull);
    MetalBindCBuffer(encoder, outCBuffer, 2ull);
    MetalBindCBuffer(encoder, inCBuffer, 3ull);

    constexpr uint64_t groupSize = 256u;
    Send<void>(encoder, "setThreadgroupMemoryLength:atIndex:", groupSize * sizeof(float), 0ull);

    const uint32_t rowCount = _out->GetDimsY() * _out->GetDimsZ();
    MTLSize threads = {rowCount * groupSize, 1, 1};
    MTLSize threadsPerThreadgroup = {groupSize, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}


template<class T>
void SplitQKV(MatrixMetal<T>* _q,
              MatrixMetal<T>* _k,
              MatrixMetal<T>* _v,
              MatrixMetal<T>* _packed)
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

    MetalManager* manager = _q->m_gInstance ? _q->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _q->Unmap();
    _k->Unmap();
    _v->Unmap();
    _packed->Unmap();

    Obj pipeline = manager->GetPipeline("split_qkv");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_q->m_dataHandles.m_cbufferHandle,
                                      _q->GetDimsX(),
                                      _q->GetDimsY(),
                                      _q->GetDimsZ(),
                                      _q->GetOffset(),
                                      static_cast<uint32_t>(_q->GetID()))
        || !manager->UpdateCachedCBuffer(_k->m_dataHandles.m_cbufferHandle,
                                         _k->GetDimsX(),
                                         _k->GetDimsY(),
                                         _k->GetDimsZ(),
                                         _k->GetOffset(),
                                         static_cast<uint32_t>(_k->GetID()))
        || !manager->UpdateCachedCBuffer(_v->m_dataHandles.m_cbufferHandle,
                                         _v->GetDimsX(),
                                         _v->GetDimsY(),
                                         _v->GetDimsZ(),
                                         _v->GetOffset(),
                                         static_cast<uint32_t>(_v->GetID()))
        || !manager->UpdateCachedCBuffer(_packed->m_dataHandles.m_cbufferHandle,
                                         _packed->GetDimsX(),
                                         _packed->GetDimsY(),
                                         _packed->GetDimsZ(),
                                         _packed->GetOffset(),
                                         static_cast<uint32_t>(_packed->GetID())))
    {
        return;
    }

    Obj qBuffer = manager->GetBuffer(_q->m_dataHandles.m_bufferHandle);
    Obj kBuffer = manager->GetBuffer(_k->m_dataHandles.m_bufferHandle);
    Obj vBuffer = manager->GetBuffer(_v->m_dataHandles.m_bufferHandle);
    Obj packedBuffer = manager->GetBuffer(_packed->m_dataHandles.m_bufferHandle);
    Obj qCBuffer = manager->GetCBuffer(_q->m_dataHandles.m_cbufferHandle);
    Obj kCBuffer = manager->GetCBuffer(_k->m_dataHandles.m_cbufferHandle);
    Obj vCBuffer = manager->GetCBuffer(_v->m_dataHandles.m_cbufferHandle);
    Obj packedCBuffer = manager->GetCBuffer(_packed->m_dataHandles.m_cbufferHandle);
    if (!qBuffer || !kBuffer || !vBuffer || !packedBuffer || !qCBuffer || !kCBuffer || !vCBuffer || !packedCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", qBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", kBuffer, 0ull, 1ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", vBuffer, 0ull, 2ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", packedBuffer, 0ull, 3ull);
    MetalBindCBuffer(encoder, qCBuffer, 4ull);
    MetalBindCBuffer(encoder, kCBuffer, 5ull);
    MetalBindCBuffer(encoder, vCBuffer, 6ull);
    MetalBindCBuffer(encoder, packedCBuffer, 7ull);

    MTLSize threads = {qCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
void MergeQKV(MatrixMetal<T>* _packedQKV,
              MatrixMetal<T>* _qHeads,
              MatrixMetal<T>* _kHeads,
              MatrixMetal<T>* _vHeads)
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

    MetalManager* manager = _packedQKV->m_gInstance ? _packedQKV->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _packedQKV->Unmap();
    _qHeads->Unmap();
    _kHeads->Unmap();
    _vHeads->Unmap();

    Obj pipeline = manager->GetPipeline("merge_qkv");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!MetalSyncCBuffer(_packedQKV) || !MetalSyncCBuffer(_qHeads) || !MetalSyncCBuffer(_kHeads)
        || !MetalSyncCBuffer(_vHeads))
    {
        return;
    }

    Obj packedBuffer = manager->GetBuffer(_packedQKV->m_dataHandles.m_bufferHandle);
    Obj qBuffer = manager->GetBuffer(_qHeads->m_dataHandles.m_bufferHandle);
    Obj kBuffer = manager->GetBuffer(_kHeads->m_dataHandles.m_bufferHandle);
    Obj vBuffer = manager->GetBuffer(_vHeads->m_dataHandles.m_bufferHandle);
    Obj packedCBuffer = manager->GetCBuffer(_packedQKV->m_dataHandles.m_cbufferHandle);
    Obj qCBuffer = manager->GetCBuffer(_qHeads->m_dataHandles.m_cbufferHandle);
    Obj kCBuffer = manager->GetCBuffer(_kHeads->m_dataHandles.m_cbufferHandle);
    Obj vCBuffer = manager->GetCBuffer(_vHeads->m_dataHandles.m_cbufferHandle);
    if (!packedBuffer || !qBuffer || !kBuffer || !vBuffer || !packedCBuffer || !qCBuffer || !kCBuffer || !vCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", packedBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", qBuffer, 0ull, 1ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", kBuffer, 0ull, 2ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", vBuffer, 0ull, 3ull);
    MetalBindCBuffer(encoder, packedCBuffer, 4ull);
    MetalBindCBuffer(encoder, qCBuffer, 5ull);
    MetalBindCBuffer(encoder, kCBuffer, 6ull);
    MetalBindCBuffer(encoder, vCBuffer, 7ull);

    MTLSize threads = {qCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}


template<class T>
void BroadcastAdd(MatrixMetal<T>* _out, MatrixMetal<T>* _L, MatrixMetal<T>* _R)
{
    assert(_out);
    assert(_L);
    assert(_R);
    assert(_out->GetDimsX() == _L->GetDimsX());
    assert(_out->GetDimsY() == _L->GetDimsY());
    assert(_out->GetDimsZ() == _L->GetDimsZ());
    assert(_R->GetElementCount() >= _L->GetDimsX());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _L->Unmap();
    _R->Unmap();

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj leftBuffer = manager->GetBuffer(_L->m_dataHandles.m_bufferHandle);
    Obj biasBuffer = manager->GetBuffer(_R->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !leftBuffer || !biasBuffer)
    {
        return;
    }

    if (outBuffer != leftBuffer)
    {
        Copy(_out, _L);
        _out->Unmap();
    }

    Obj pipeline = manager->GetPipeline("broadcast_add");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID()))
        || !manager->UpdateCachedCBuffer(_R->m_dataHandles.m_cbufferHandle,
                                         _R->GetDimsX(),
                                         _R->GetDimsY(),
                                         _R->GetDimsZ(),
                                         _R->GetOffset(),
                                         static_cast<uint32_t>(_R->GetID())))
    {
        return;
    }

    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj biasCBuffer = manager->GetCBuffer(_R->m_dataHandles.m_cbufferHandle);
    if (!outCBuffer || !biasCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", biasBuffer, 0ull, 1ull);
    MetalBindCBuffer(encoder, outCBuffer, 2ull);
    MetalBindCBuffer(encoder, biasCBuffer, 3ull);

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
void LayerNormOp(MatrixMetal<T>* _out,
                 MatrixMetal<T>* _in,
                 MatrixMetal<T>* _gamma,
                 MatrixMetal<T>* _beta = nullptr,
                 MatrixMetal<T>* _xHat = nullptr,
                 T _eps = static_cast<T>(1e-5),
                 MatrixMetal<T>* _invStd = nullptr)
{
    assert(_out);
    assert(_in);
    assert(_gamma);
    if (_invStd)
    {
        assert(_xHat);
        assert(_invStd->GetDimsX() == 1u);
        assert(_invStd->GetDimsY() == _in->GetDimsY());
        assert(_invStd->GetDimsZ() == _in->GetDimsZ());
    }

    assert(_out->GetDimsX() == _in->GetDimsX());
    assert(_out->GetDimsY() == _in->GetDimsY());
    assert(_out->GetDimsZ() == _in->GetDimsZ());
    assert(_gamma->GetDimsX() == _in->GetDimsX());
    assert(_gamma->GetElementCount() >= _in->GetDimsX());
    if (_beta)
    {
        assert(_beta->GetDimsX() == _in->GetDimsX());
        assert(_beta->GetElementCount() >= _in->GetDimsX());
    }
    if (_xHat)
    {
        assert(_xHat->GetDimsX() == _in->GetDimsX());
        assert(_xHat->GetDimsY() == _in->GetDimsY());
        assert(_xHat->GetDimsZ() == _in->GetDimsZ());
    }
    (void)_eps;
    if (_invStd) _invStd->Unmap();

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

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

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj inBuffer = manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    Obj gammaBuffer = manager->GetBuffer(_gamma->m_dataHandles.m_bufferHandle);
    Obj betaBuffer = _beta ? manager->GetBuffer(_beta->m_dataHandles.m_bufferHandle) : nullptr;
    Obj xHatBuffer = _xHat ? manager->GetBuffer(_xHat->m_dataHandles.m_bufferHandle) : nullptr;
    if (!outBuffer || !inBuffer || !gammaBuffer || (_beta && !betaBuffer) || (_xHat && !xHatBuffer))
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
        shaderName = _invStd ? (hasBeta ? "layernorm_cached" : "layernorm_nobeta_cached")
                             : (hasBeta ? "layernorm" : "layernorm_nobeta");
    }
    else
    {
        shaderName = hasBeta ? "layernorm_no_xhat" : "layernorm_nobeta_no_xhat";
    }

    Obj pipeline = manager->GetPipeline(shaderName);
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID()))
        || !manager->UpdateCachedCBuffer(_in->m_dataHandles.m_cbufferHandle,
                                         _in->GetDimsX(),
                                         _in->GetDimsY(),
                                         _in->GetDimsZ(),
                                         _in->GetOffset(),
                                         static_cast<uint32_t>(_in->GetID()))
        || !manager->UpdateCachedCBuffer(_gamma->m_dataHandles.m_cbufferHandle,
                                         _gamma->GetDimsX(),
                                         _gamma->GetDimsY(),
                                         _gamma->GetDimsZ(),
                                         _gamma->GetOffset(),
                                         static_cast<uint32_t>(_gamma->GetID())))
    {
        return;
    }
    if (hasBeta
        && !manager->UpdateCachedCBuffer(_beta->m_dataHandles.m_cbufferHandle,
                                         _beta->GetDimsX(),
                                         _beta->GetDimsY(),
                                         _beta->GetDimsZ(),
                                         _beta->GetOffset(),
                                         static_cast<uint32_t>(_beta->GetID())))
    {
        return;
    }
    if (hasXHat
        && !manager->UpdateCachedCBuffer(_xHat->m_dataHandles.m_cbufferHandle,
                                         _xHat->GetDimsX(),
                                         _xHat->GetDimsY(),
                                         _xHat->GetDimsZ(),
                                         _xHat->GetOffset(),
                                         static_cast<uint32_t>(_xHat->GetID())))
    {
        return;
    }

    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj inCBuffer = manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    Obj gammaCBuffer = manager->GetCBuffer(_gamma->m_dataHandles.m_cbufferHandle);
    Obj betaCBuffer = hasBeta ? manager->GetCBuffer(_beta->m_dataHandles.m_cbufferHandle) : nullptr;
    Obj xHatCBuffer = hasXHat ? manager->GetCBuffer(_xHat->m_dataHandles.m_cbufferHandle) : nullptr;
    if (!outCBuffer || !inCBuffer || !gammaCBuffer || (hasBeta && !betaCBuffer) || (hasXHat && !xHatCBuffer))
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    if (hasXHat)
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", xHatBuffer, 0ull, 1ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 2ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", gammaBuffer, 0ull, 3ull);
        if (hasBeta)
        {
            Send<void>(encoder, "setBuffer:offset:atIndex:", betaBuffer, 0ull, 4ull);
            MetalBindCBuffer(encoder, outCBuffer, 5ull);
            MetalBindCBuffer(encoder, inCBuffer, 6ull);
            MetalBindCBuffer(encoder, gammaCBuffer, 7ull);
            MetalBindCBuffer(encoder, betaCBuffer, 8ull);
            MetalBindCBuffer(encoder, xHatCBuffer, 9ull);
        }
        else
        {
            MetalBindCBuffer(encoder, outCBuffer, 4ull);
            MetalBindCBuffer(encoder, inCBuffer, 5ull);
            MetalBindCBuffer(encoder, gammaCBuffer, 6ull);
            MetalBindCBuffer(encoder, xHatCBuffer, 7ull);
        }
    }
    else
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 1ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", gammaBuffer, 0ull, 2ull);
        if (hasBeta)
        {
            Send<void>(encoder, "setBuffer:offset:atIndex:", betaBuffer, 0ull, 3ull);
            MetalBindCBuffer(encoder, outCBuffer, 4ull);
            MetalBindCBuffer(encoder, inCBuffer, 5ull);
            MetalBindCBuffer(encoder, gammaCBuffer, 6ull);
            MetalBindCBuffer(encoder, betaCBuffer, 7ull);
        }
        else
        {
            MetalBindCBuffer(encoder, outCBuffer, 3ull);
            MetalBindCBuffer(encoder, inCBuffer, 4ull);
            MetalBindCBuffer(encoder, gammaCBuffer, 5ull);
        }
    }

    if (_invStd)
    {
        if (!MetalSyncCBuffer(_invStd)) return;
        const uint64_t slot = hasBeta ? 10ull : 8ull;
        Send<void>(encoder, "setBuffer:offset:atIndex:", manager->GetBuffer(_invStd->m_dataHandles.m_bufferHandle), 0ull, slot);
        MetalBindCBuffer(encoder, manager->GetCBuffer(_invStd->m_dataHandles.m_cbufferHandle), slot + 1ull);
    }

    constexpr uint64_t groupSize = 256u;
    Send<void>(encoder, "setThreadgroupMemoryLength:atIndex:", groupSize * sizeof(float), 0ull);

    const uint32_t rowCount = _out->GetDimsY() * _out->GetDimsZ();
    MTLSize threads = {rowCount * groupSize, 1, 1};
    MTLSize threadsPerThreadgroup = {groupSize, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
void LayerNormBackwardsOp(MatrixMetal<T>* _dX,
                          MatrixMetal<T>* _dY,
                          MatrixMetal<T>* _in,
                          MatrixMetal<T>* _gamma,
                          MatrixMetal<T>* _dGamma,
                          MatrixMetal<T>* _dBeta = nullptr,
                          MatrixMetal<T>* _xHat = nullptr,
                          T _eps = static_cast<T>(1e-5),
                          MatrixMetal<T>* _invStd = nullptr)
{
    assert(_dX);
    assert(_dY);
    assert(_in);
    assert(_gamma);
    if (_invStd)
    {
        assert(_xHat);
        assert(_invStd->GetDimsX() == 1u);
        assert(_invStd->GetDimsY() == _in->GetDimsY());
        assert(_invStd->GetDimsZ() == _in->GetDimsZ());
    }

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
    if (_invStd) _invStd->Unmap();

    if (!_xHat)
    {
        return;
    }

    MetalManager* manager = _dX->m_gInstance ? _dX->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _dX->Unmap();
    _dY->Unmap();
    if (!_invStd) _in->Unmap();
    _gamma->Unmap();
    _dGamma->Unmap();
    _xHat->Unmap();
    if (_dBeta)
    {
        _dBeta->Unmap();
    }

    const bool hasBeta = (_dBeta != nullptr);
    Obj parallelParamsPipeline = _dY->GetDimsY()*_dY->GetDimsZ() >= 64u
        ? manager->GetPipeline(hasBeta ? "layernorm_backwards_params_parallel" : "layernorm_backwards_params_nobeta_parallel") : nullptr;
    const bool parallelParams = parallelParamsPipeline != nullptr;
    Obj paramsPipeline = parallelParams ? parallelParamsPipeline
        : manager->GetPipeline(hasBeta ? "layernorm_backwards_params" : "layernorm_backwards_params_nobeta");
    Obj dxPipeline = manager->GetPipeline(_invStd ? "layernorm_backwards_dx_cached" : "layernorm_backwards_dx");
    Obj queue = manager->GetQueue();
    if (!paramsPipeline || !dxPipeline || !queue)
    {
        return;
    }

    if (!MetalSyncCBuffer(_dX) || !MetalSyncCBuffer(_dY) || !MetalSyncCBuffer(_in) || !MetalSyncCBuffer(_gamma)
        || !MetalSyncCBuffer(_dGamma) || !MetalSyncCBuffer(_xHat))
    {
        return;
    }
    if (hasBeta && !MetalSyncCBuffer(_dBeta))
    {
        return;
    }

    Obj dXBuffer = manager->GetBuffer(_dX->m_dataHandles.m_bufferHandle);
    Obj dYBuffer = manager->GetBuffer(_dY->m_dataHandles.m_bufferHandle);
    Obj inBuffer = manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    Obj gammaBuffer = manager->GetBuffer(_gamma->m_dataHandles.m_bufferHandle);
    Obj dGammaBuffer = manager->GetBuffer(_dGamma->m_dataHandles.m_bufferHandle);
    Obj dBetaBuffer = hasBeta ? manager->GetBuffer(_dBeta->m_dataHandles.m_bufferHandle) : nullptr;
    Obj xHatBuffer = manager->GetBuffer(_xHat->m_dataHandles.m_bufferHandle);
    if (!dXBuffer || !dYBuffer || !inBuffer || !gammaBuffer || !dGammaBuffer || !xHatBuffer || (hasBeta && !dBetaBuffer))
    {
        return;
    }

    Obj dXCBuffer = manager->GetCBuffer(_dX->m_dataHandles.m_cbufferHandle);
    Obj dYCBuffer = manager->GetCBuffer(_dY->m_dataHandles.m_cbufferHandle);
    Obj inCBuffer = manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    Obj gammaCBuffer = manager->GetCBuffer(_gamma->m_dataHandles.m_cbufferHandle);
    Obj dGammaCBuffer = manager->GetCBuffer(_dGamma->m_dataHandles.m_cbufferHandle);
    Obj dBetaCBuffer = hasBeta ? manager->GetCBuffer(_dBeta->m_dataHandles.m_cbufferHandle) : nullptr;
    Obj xHatCBuffer = manager->GetCBuffer(_xHat->m_dataHandles.m_cbufferHandle);
    if (!dXCBuffer || !dYCBuffer || !inCBuffer || !gammaCBuffer || !dGammaCBuffer || !xHatCBuffer
        || (hasBeta && !dBetaCBuffer))
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", paramsPipeline);
    if (hasBeta)
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", dGammaBuffer, 0ull, 0ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", dBetaBuffer, 0ull, 1ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", dYBuffer, 0ull, 2ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", xHatBuffer, 0ull, 3ull);
        MetalBindCBuffer(encoder, dGammaCBuffer, 4ull);
        MetalBindCBuffer(encoder, dBetaCBuffer, 5ull);
        MetalBindCBuffer(encoder, dYCBuffer, 6ull);
        MetalBindCBuffer(encoder, xHatCBuffer, 7ull);
    }
    else
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", dGammaBuffer, 0ull, 0ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", dYBuffer, 0ull, 1ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", xHatBuffer, 0ull, 2ull);
        MetalBindCBuffer(encoder, dGammaCBuffer, 3ull);
        MetalBindCBuffer(encoder, dYCBuffer, 4ull);
        MetalBindCBuffer(encoder, xHatCBuffer, 5ull);
    }

    MTLSize paramThreads = {_dGamma->GetDimsX(), 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    if (parallelParams)
        Send<void>(encoder, "dispatchThreadgroups:threadsPerThreadgroup:",
            MTLSize{(_dGamma->GetDimsX()+31u)/32u,1u,1u},MTLSize{32u,8u,1u});
    else
        Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", paramThreads, threadsPerThreadgroup);

    Send<void>(encoder, "setComputePipelineState:", dxPipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", dXBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", dYBuffer, 0ull, 1ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", _invStd ? xHatBuffer : inBuffer, 0ull, 2ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", gammaBuffer, 0ull, 3ull);
    MetalBindCBuffer(encoder, dXCBuffer, 4ull);
    MetalBindCBuffer(encoder, dYCBuffer, 5ull);
    MetalBindCBuffer(encoder, _invStd ? xHatCBuffer : inCBuffer, 6ull);
    MetalBindCBuffer(encoder, gammaCBuffer, 7ull);

    if (_invStd)
    {
        if (!MetalSyncCBuffer(_invStd)) return;
        Send<void>(encoder, "setBuffer:offset:atIndex:", manager->GetBuffer(_invStd->m_dataHandles.m_bufferHandle), 0ull, 8ull);
        MetalBindCBuffer(encoder, manager->GetCBuffer(_invStd->m_dataHandles.m_cbufferHandle), 9ull);
    }

    constexpr uint64_t groupSize = 256u;
    Send<void>(encoder, "setThreadgroupMemoryLength:atIndex:", groupSize * sizeof(float), 0ull);
    Send<void>(encoder, "setThreadgroupMemoryLength:atIndex:", groupSize * sizeof(float), 1ull);
    if (!_invStd)
        Send<void>(encoder, "setThreadgroupMemoryLength:atIndex:", groupSize * sizeof(float), 2ull);

    const uint32_t rowCount = _dX->GetDimsY() * _dX->GetDimsZ();
    MTLSize dxThreads = {rowCount * groupSize, 1, 1};
    MTLSize dxThreadsPerThreadgroup = {groupSize, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", dxThreads, dxThreadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
void PerElementMul(MatrixMetal<T>* _out, MatrixMetal<T>* _in, MatrixMetal<T>* _mask)
{
    assert(_out);
    assert(_in);
    assert(_mask);
    assert(_out->GetElementCount() == _in->GetElementCount());
    assert(_out->GetElementCount() == _mask->GetElementCount());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _in->Unmap();
    _mask->Unmap();

    const bool aliasOutIn = _out->m_dataHandles.m_bufferHandle == _in->m_dataHandles.m_bufferHandle;
    Obj pipeline = manager->GetPipeline(aliasOutIn ? "per_element_mul_inplace" : "per_element_mul");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID()))
        || !manager->UpdateCachedCBuffer(_mask->m_dataHandles.m_cbufferHandle,
                                         _mask->GetDimsX(),
                                         _mask->GetDimsY(),
                                         _mask->GetDimsZ(),
                                         _mask->GetOffset(),
                                         static_cast<uint32_t>(_mask->GetID()))
        || (!aliasOutIn
            && !manager->UpdateCachedCBuffer(_in->m_dataHandles.m_cbufferHandle,
                                             _in->GetDimsX(),
                                             _in->GetDimsY(),
                                             _in->GetDimsZ(),
                                             _in->GetOffset(),
                                             static_cast<uint32_t>(_in->GetID()))))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj maskBuffer = manager->GetBuffer(_mask->m_dataHandles.m_bufferHandle);
    Obj inBuffer = aliasOutIn ? nullptr : manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj maskCBuffer = manager->GetCBuffer(_mask->m_dataHandles.m_cbufferHandle);
    Obj inCBuffer = aliasOutIn ? nullptr : manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !maskBuffer || !outCBuffer || !maskCBuffer || (!aliasOutIn && (!inBuffer || !inCBuffer)))
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    if (aliasOutIn)
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", maskBuffer, 0ull, 1ull);
        MetalBindCBuffer(encoder, outCBuffer, 2ull);
        MetalBindCBuffer(encoder, maskCBuffer, 3ull);
    }
    else
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 1ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", maskBuffer, 0ull, 2ull);
        MetalBindCBuffer(encoder, outCBuffer, 3ull);
        MetalBindCBuffer(encoder, inCBuffer, 4ull);
        MetalBindCBuffer(encoder, maskCBuffer, 5ull);
    }

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
static void DispatchUnaryMetalOp(MatrixMetal<T>* _out,
                                 MatrixMetal<T>* _in,
                                 const char* shaderName,
                                 const char* inplaceShaderName)
{
    assert(_out);
    assert(_in);
    assert(_out->GetElementCount() == _in->GetElementCount());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _in->Unmap();

    const bool aliasInOut = _out->m_dataHandles.m_bufferHandle == _in->m_dataHandles.m_bufferHandle;
    Obj pipeline = manager->GetPipeline(aliasInOut ? inplaceShaderName : shaderName);
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!MetalSyncCBuffer(_out) || (!aliasInOut && !MetalSyncCBuffer(_in)))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj inBuffer = aliasInOut ? nullptr : manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj inCBuffer = aliasInOut ? nullptr : manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !outCBuffer || (!aliasInOut && (!inBuffer || !inCBuffer)))
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    if (aliasInOut)
    {
        MetalBindCBuffer(encoder, outCBuffer, 1ull);
    }
    else
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 1ull);
        MetalBindCBuffer(encoder, outCBuffer, 2ull);
        MetalBindCBuffer(encoder, inCBuffer, 3ull);
    }

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
void GeluMat(MatrixMetal<T>* _out, MatrixMetal<T>* _in)
{
    DispatchUnaryMetalOp(_out, _in, "gelu", "gelu_inplace");
}

template<class T>
void GeluDerivtiveMat(MatrixMetal<T>* _out, MatrixMetal<T>* _in)
{
    DispatchUnaryMetalOp(_out, _in, "gelu_derivative", "gelu_derivative_inplace");
}


template<class T>
void GeluBackwards(MatrixMetal<T>* out, MatrixMetal<T>* input, MatrixMetal<T>* grad)
{
    assert(out && input && grad);
    assert(out->GetDimsX() == input->GetDimsX() && out->GetDimsY() == input->GetDimsY()
           && out->GetDimsZ() == input->GetDimsZ());
    assert(out->GetDimsX() == grad->GetDimsX() && out->GetDimsY() == grad->GetDimsY()
           && out->GetDimsZ() == grad->GetDimsZ());
    if (out->GetElementCount() == 0u) return;
    out->Unmap(); input->Unmap(); grad->Unmap();
    auto* manager = out->m_gInstance;
    if (!MetalSyncCBuffer(out) || !MetalSyncCBuffer(input) || !MetalSyncCBuffer(grad)) return;
    Obj pipeline = manager->GetPipeline("gelu_backwards");
    Obj encoder = manager->GetEncoder();
    if (!pipeline || !encoder) return;
    Send<void>(encoder, "setComputePipelineState:", pipeline);
    MatrixMetal<T>* matrices[] = {out, input, grad};
    for (uint64_t i = 0; i < 3; ++i)
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:",
                   manager->GetBuffer(matrices[i]->m_dataHandles.m_bufferHandle), 0ull, i);
        MetalBindCBuffer(encoder, manager->GetCBuffer(matrices[i]->m_dataHandles.m_cbufferHandle), i + 3ull);
    }
    MTLSize threads = {out->GetElementCount(), 1, 1};
    MTLSize group = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, group);
    manager->WorkSubmitted();
}

template<class T>
static inline uint32_t MetalPackFloatBits(T value)
{
    static_assert(sizeof(T) == sizeof(uint32_t), "MetalPackFloatBits expects a 32-bit value.");
    uint32_t packed = 0u;
    std::memcpy(&packed, &value, sizeof(uint32_t));
    return packed;
}

template<class T>
void Scale(MatrixMetal<T>* _out, MatrixMetal<T>* _in, MatrixMetal<T>* _scale)
{
    assert(_out);
    assert(_in);
    assert(_scale);
    assert(_out->GetElementCount() == _in->GetElementCount());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _in->Unmap();
    _scale->Unmap();

    const bool aliasInOut = _out->m_dataHandles.m_bufferHandle == _in->m_dataHandles.m_bufferHandle;
    const char* shaderName = aliasInOut ? "scale_inplace" : "scale";
    Obj pipeline = manager->GetPipeline(shaderName);
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID()))
        || !manager->UpdateCachedCBuffer(_scale->m_dataHandles.m_cbufferHandle,
                                         _scale->GetDimsX(),
                                         _scale->GetDimsY(),
                                         _scale->GetDimsZ(),
                                         _scale->GetOffset(),
                                         static_cast<uint32_t>(_scale->GetID()))
        || (!aliasInOut
            && !manager->UpdateCachedCBuffer(_in->m_dataHandles.m_cbufferHandle,
                                             _in->GetDimsX(),
                                             _in->GetDimsY(),
                                             _in->GetDimsZ(),
                                             _in->GetOffset(),
                                             static_cast<uint32_t>(_in->GetID()))))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj scaleBuffer = manager->GetBuffer(_scale->m_dataHandles.m_bufferHandle);
    Obj inBuffer = aliasInOut ? nullptr : manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj scaleCBuffer = manager->GetCBuffer(_scale->m_dataHandles.m_cbufferHandle);
    Obj inCBuffer = aliasInOut ? nullptr : manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !scaleBuffer || !outCBuffer || !scaleCBuffer || (!aliasInOut && (!inBuffer || !inCBuffer)))
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    if (aliasInOut)
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", scaleBuffer, 0ull, 1ull);
        MetalBindCBuffer(encoder, outCBuffer, 2ull);
        MetalBindCBuffer(encoder, scaleCBuffer, 3ull);
    }
    else
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 1ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", scaleBuffer, 0ull, 2ull);
        MetalBindCBuffer(encoder, outCBuffer, 3ull);
        MetalBindCBuffer(encoder, inCBuffer, 4ull);
        MetalBindCBuffer(encoder, scaleCBuffer, 5ull);
    }

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
void Scale(MatrixMetal<T>* _out, MatrixMetal<T>* _in, float _scale)
{
    assert(_out);
    assert(_in);
    assert(_out->GetElementCount() == _in->GetElementCount());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _in->Unmap();

    const bool aliasInOut = _out->m_dataHandles.m_bufferHandle == _in->m_dataHandles.m_bufferHandle;
    const char* shaderName = aliasInOut ? "scale_scalar_inplace" : "scale_scalar";
    Obj pipeline = manager->GetPipeline(shaderName);
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID()))
        || (!aliasInOut
            && !manager->UpdateCachedCBuffer(_in->m_dataHandles.m_cbufferHandle,
                                             _in->GetDimsX(),
                                             _in->GetDimsY(),
                                             _in->GetDimsZ(),
                                             _in->GetOffset(),
                                             static_cast<uint32_t>(_in->GetID()))))
    {
        return;
    }

    const uint32_t optionalParams[1] = {MetalPackFloatBits(_scale)};
    if (!manager->SetCBufferOptionalParams(_out->m_dataHandles.m_cbufferHandle, optionalParams, 1u))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj inBuffer = aliasInOut ? nullptr : manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj inCBuffer = aliasInOut ? nullptr : manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !outCBuffer || (!aliasInOut && (!inBuffer || !inCBuffer)))
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    if (aliasInOut)
    {
        MetalBindCBuffer(encoder, outCBuffer, 1ull);
    }
    else
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 1ull);
        MetalBindCBuffer(encoder, outCBuffer, 2ull);
        MetalBindCBuffer(encoder, inCBuffer, 3ull);
    }

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
void Fill(MatrixMetal<T>* _out, MatrixMetal<T>* _val)
{
    assert(_out);
    assert(_val);
    assert(_val->GetElementCount() >= 1u);

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _val->Unmap();

    Obj pipeline = manager->GetPipeline("fill");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID()))
        || !manager->UpdateCachedCBuffer(_val->m_dataHandles.m_cbufferHandle,
                                         _val->GetDimsX(),
                                         _val->GetDimsY(),
                                         _val->GetDimsZ(),
                                         _val->GetOffset(),
                                         static_cast<uint32_t>(_val->GetID())))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj valBuffer = manager->GetBuffer(_val->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj valCBuffer = manager->GetCBuffer(_val->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !valBuffer || !outCBuffer || !valCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", valBuffer, 0ull, 1ull);
    MetalBindCBuffer(encoder, outCBuffer, 2ull);
    MetalBindCBuffer(encoder, valCBuffer, 3ull);

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
void DropoutMask(MatrixMetal<T>* _out, float probability,
                 uint32_t seed = RandomUtils::NextSeed())
{
    assert(_out && std::isfinite(probability));
    if (_out->GetElementCount() == 0u) return;

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();

    Obj pipeline = manager->GetPipeline("dropout_mask");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID())))
    {
        return;
    }

    const uint32_t optionalParams[2] = {MetalPackFloatBits(probability), seed};
    if (!manager->SetCBufferOptionalParams(_out->m_dataHandles.m_cbufferHandle, optionalParams, 2u))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !outCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    MetalBindCBuffer(encoder, outCBuffer, 1ull);

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}


template<class T>
void Fill(MatrixMetal<T>* _out, float _val)
{
    assert(_out);

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();

    Obj pipeline = manager->GetPipeline("fill_scalar");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID())))
    {
        return;
    }

    const uint32_t optionalParams[1] = {MetalPackFloatBits(_val)};
    if (!manager->SetCBufferOptionalParams(_out->m_dataHandles.m_cbufferHandle, optionalParams, 1u))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !outCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    MetalBindCBuffer(encoder, outCBuffer, 1ull);

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}


template<class T>
void AdamWUpdate(MatrixMetal<T>* param,
                 MatrixMetal<T>* grad,
                 MatrixMetal<T>* mt,
                 MatrixMetal<T>* vt,
                 T lr,
                 T beta1,
                 T beta2,
                 T beta1_pow_t,
                 T beta2_pow_t,
                 T eps,
                 T weightDecay)
{
    if (!param || !grad || !mt || !vt)
    {
        return;
    }

    MetalManager* manager = param->m_gInstance ? param->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
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

    Obj pipeline = manager->GetPipeline("adam_update");
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!MetalSyncCBuffer(param) || !MetalSyncCBuffer(grad) || !MetalSyncCBuffer(mt) || !MetalSyncCBuffer(vt))
    {
        return;
    }

    const uint32_t optionalParams[7] =
    {
        MetalPackFloatBits(lr),
        MetalPackFloatBits(beta1),
        MetalPackFloatBits(beta2),
        MetalPackFloatBits(beta1_pow_t),
        MetalPackFloatBits(beta2_pow_t),
        MetalPackFloatBits(eps),
        MetalPackFloatBits(weightDecay)
    };
    if (!manager->SetCBufferOptionalParams(param->m_dataHandles.m_cbufferHandle, optionalParams, 7u))
    {
        return;
    }

    Obj paramBuffer = manager->GetBuffer(param->m_dataHandles.m_bufferHandle);
    Obj gradBuffer = manager->GetBuffer(grad->m_dataHandles.m_bufferHandle);
    Obj mtBuffer = manager->GetBuffer(mt->m_dataHandles.m_bufferHandle);
    Obj vtBuffer = manager->GetBuffer(vt->m_dataHandles.m_bufferHandle);
    if (!paramBuffer || !gradBuffer || !mtBuffer || !vtBuffer)
    {
        return;
    }

    if (paramBuffer == gradBuffer || paramBuffer == mtBuffer || paramBuffer == vtBuffer
        || gradBuffer == mtBuffer || gradBuffer == vtBuffer || mtBuffer == vtBuffer)
    {
        return;
    }

    Obj paramCBuffer = manager->GetCBuffer(param->m_dataHandles.m_cbufferHandle);
    Obj gradCBuffer = manager->GetCBuffer(grad->m_dataHandles.m_cbufferHandle);
    Obj mtCBuffer = manager->GetCBuffer(mt->m_dataHandles.m_cbufferHandle);
    Obj vtCBuffer = manager->GetCBuffer(vt->m_dataHandles.m_cbufferHandle);
    if (!paramCBuffer || !gradCBuffer || !mtCBuffer || !vtCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", paramBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", mtBuffer, 0ull, 1ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", vtBuffer, 0ull, 2ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", gradBuffer, 0ull, 3ull);
    MetalBindCBuffer(encoder, paramCBuffer, 4ull);
    MetalBindCBuffer(encoder, gradCBuffer, 5ull);
    MetalBindCBuffer(encoder, mtCBuffer, 6ull);
    MetalBindCBuffer(encoder, vtCBuffer, 7ull);

    const uint32_t elementCount = param->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}

template<class T>
void AdamUpdate(MatrixMetal<T>* param,
                MatrixMetal<T>* grad,
                MatrixMetal<T>* mt,
                MatrixMetal<T>* vt,
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




template<class T>
void ScaleAdd(MatrixMetal<T>* _out, MatrixMetal<T>* _in, MatrixMetal<T>* _scale, MatrixMetal<T>* _add)
{
    assert(_out);
    assert(_in);
    assert(_scale);
    assert(_add);
    assert(_out->GetElementCount() == _in->GetElementCount());
    assert(_out->GetElementCount() == _add->GetElementCount());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _in->Unmap();
    _scale->Unmap();
    _add->Unmap();

    const bool aliasOutIn = _out->m_dataHandles.m_bufferHandle == _in->m_dataHandles.m_bufferHandle;
    const char* shaderName = aliasOutIn ? "scaleadd_inplace" : "scaleadd";
    Obj pipeline = manager->GetPipeline(shaderName);
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID()))
        || !manager->UpdateCachedCBuffer(_scale->m_dataHandles.m_cbufferHandle,
                                         _scale->GetDimsX(),
                                         _scale->GetDimsY(),
                                         _scale->GetDimsZ(),
                                         _scale->GetOffset(),
                                         static_cast<uint32_t>(_scale->GetID()))
        || !manager->UpdateCachedCBuffer(_add->m_dataHandles.m_cbufferHandle,
                                         _add->GetDimsX(),
                                         _add->GetDimsY(),
                                         _add->GetDimsZ(),
                                         _add->GetOffset(),
                                         static_cast<uint32_t>(_add->GetID()))
        || (!aliasOutIn
            && !manager->UpdateCachedCBuffer(_in->m_dataHandles.m_cbufferHandle,
                                             _in->GetDimsX(),
                                             _in->GetDimsY(),
                                             _in->GetDimsZ(),
                                             _in->GetOffset(),
                                             static_cast<uint32_t>(_in->GetID()))))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj scaleBuffer = manager->GetBuffer(_scale->m_dataHandles.m_bufferHandle);
    Obj addBuffer = manager->GetBuffer(_add->m_dataHandles.m_bufferHandle);
    Obj inBuffer = aliasOutIn ? nullptr : manager->GetBuffer(_in->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj scaleCBuffer = manager->GetCBuffer(_scale->m_dataHandles.m_cbufferHandle);
    Obj addCBuffer = manager->GetCBuffer(_add->m_dataHandles.m_cbufferHandle);
    Obj inCBuffer = aliasOutIn ? nullptr : manager->GetCBuffer(_in->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !scaleBuffer || !addBuffer || !outCBuffer || !scaleCBuffer || !addCBuffer
        || (!aliasOutIn && (!inBuffer || !inCBuffer)))
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    if (aliasOutIn)
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", scaleBuffer, 0ull, 1ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", addBuffer, 0ull, 2ull);
        MetalBindCBuffer(encoder, outCBuffer, 3ull);
        MetalBindCBuffer(encoder, scaleCBuffer, 4ull);
        MetalBindCBuffer(encoder, addCBuffer, 5ull);
    }
    else
    {
        Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 1ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", scaleBuffer, 0ull, 2ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", addBuffer, 0ull, 3ull);
        MetalBindCBuffer(encoder, outCBuffer, 4ull);
        MetalBindCBuffer(encoder, inCBuffer, 5ull);
        MetalBindCBuffer(encoder, scaleCBuffer, 6ull);
        MetalBindCBuffer(encoder, addCBuffer, 7ull);
    }

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}


template<class T>
void Add(MatrixMetal<T>* _out, MatrixMetal<T>* _L, MatrixMetal<T>* _R)
{
    assert(_out);
    assert(_L);
    assert(_R);
    assert(_out->GetElementCount() == _L->GetElementCount());
    assert(_out->GetElementCount() == _R->GetElementCount());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _L->Unmap();
    _R->Unmap();

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj leftBuffer = manager->GetBuffer(_L->m_dataHandles.m_bufferHandle);
    Obj rightBuffer = manager->GetBuffer(_R->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !leftBuffer || !rightBuffer)
    {
        return;
    }

    const bool sameOutLeftBuffer = _out->m_dataHandles.m_bufferHandle == _L->m_dataHandles.m_bufferHandle;
    const bool sameOutRightBuffer = _out->m_dataHandles.m_bufferHandle == _R->m_dataHandles.m_bufferHandle;
    const bool aliasOutLeft = sameOutLeftBuffer && (_out->GetOffset() == _L->GetOffset());
    const bool aliasOutRight = sameOutRightBuffer && (_out->GetOffset() == _R->GetOffset());
    if (aliasOutLeft && aliasOutRight)
    {
        return;
    }

    const bool canUseMetalInPlace = (aliasOutLeft && !sameOutRightBuffer)
                                    || (aliasOutRight && !sameOutLeftBuffer);
    const bool outputSharesInputBacking = sameOutLeftBuffer || sameOutRightBuffer;
    if (outputSharesInputBacking && !canUseMetalInPlace)
    {
        const uint32_t elementCount = _out->GetElementCount();
        std::vector<T> left(elementCount);
        std::vector<T> right(elementCount);
        std::memcpy(left.data(), _L->DataRead(), sizeof(T) * elementCount);
        std::memcpy(right.data(), _R->DataRead(), sizeof(T) * elementCount);

        T* outPtr = _out->DataWrite();
        for (uint32_t i = 0; i < elementCount; ++i)
        {
            outPtr[i] = left[i] + right[i];
        }
        _out->Unmap();
        manager->DumpInfoQueueErrors();
        return;
    }

    const char* shaderName = (aliasOutLeft || aliasOutRight) ? "add_inplace" : "add";
    Obj pipeline = manager->GetPipeline(shaderName);
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    if (aliasOutLeft || aliasOutRight)
    {
        MatrixMetal<T>* in = aliasOutLeft ? _R : _L;
        if (!MetalSyncCBuffer(_out) || !MetalSyncCBuffer(in))
        {
            return;
        }

        Obj inBuffer = manager->GetBuffer(in->m_dataHandles.m_bufferHandle);
        Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
        Obj inCBuffer = manager->GetCBuffer(in->m_dataHandles.m_cbufferHandle);
        if (!inBuffer || !outCBuffer || !inCBuffer)
        {
            return;
        }

        Send<void>(encoder, "setComputePipelineState:", pipeline);
        Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 1ull);
        MetalBindCBuffer(encoder, outCBuffer, 2ull);
        MetalBindCBuffer(encoder, inCBuffer, 3ull);
    }
    else
    {
        if (!MetalSyncCBuffer(_out) || !MetalSyncCBuffer(_L) || !MetalSyncCBuffer(_R))
        {
            return;
        }

        Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
        Obj leftCBuffer = manager->GetCBuffer(_L->m_dataHandles.m_cbufferHandle);
        Obj rightCBuffer = manager->GetCBuffer(_R->m_dataHandles.m_cbufferHandle);
        if (!outCBuffer || !leftCBuffer || !rightCBuffer)
        {
            return;
        }

        Send<void>(encoder, "setComputePipelineState:", pipeline);
        Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", leftBuffer, 0ull, 1ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", rightBuffer, 0ull, 2ull);
        MetalBindCBuffer(encoder, outCBuffer, 3ull);
        MetalBindCBuffer(encoder, leftCBuffer, 4ull);
        MetalBindCBuffer(encoder, rightCBuffer, 5ull);
    }

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}



template<class T>
void Sub(MatrixMetal<T>* _out, MatrixMetal<T>* _L, MatrixMetal<T>* _R)
{
    assert(_out);
    assert(_L);
    assert(_R);
    assert(_out->GetElementCount() == _L->GetElementCount());
    assert(_out->GetElementCount() == _R->GetElementCount());

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _L->Unmap();
    _R->Unmap();

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj leftBuffer = manager->GetBuffer(_L->m_dataHandles.m_bufferHandle);
    Obj rightBuffer = manager->GetBuffer(_R->m_dataHandles.m_bufferHandle);
    if (!outBuffer || !leftBuffer || !rightBuffer)
    {
        return;
    }

    const bool aliasOutLeft = (outBuffer == leftBuffer);
    const bool aliasOutRight = (outBuffer == rightBuffer);
    if (aliasOutLeft && aliasOutRight)
    {
        Clear(_out);
        return;
    }

    const char* shaderName = (aliasOutLeft || aliasOutRight) ? "sub_inplace" : "sub";
    Obj pipeline = manager->GetPipeline(shaderName);
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    if (aliasOutLeft || aliasOutRight)
    {
        MatrixMetal<T>* in = aliasOutLeft ? _R : _L;
        if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                          _out->GetDimsX(),
                                          _out->GetDimsY(),
                                          _out->GetDimsZ(),
                                          _out->GetOffset(),
                                          static_cast<uint32_t>(_out->GetID()))
            || !manager->UpdateCachedCBuffer(in->m_dataHandles.m_cbufferHandle,
                                             in->GetDimsX(),
                                             in->GetDimsY(),
                                             in->GetDimsZ(),
                                             in->GetOffset(),
                                             static_cast<uint32_t>(in->GetID())))
        {
            return;
        }

        const uint32_t optionalParams[1] = { aliasOutRight ? 1u : 0u };
        if (!manager->SetCBufferOptionalParams(_out->m_dataHandles.m_cbufferHandle, optionalParams, 1u))
        {
            return;
        }

        Obj inBuffer = manager->GetBuffer(in->m_dataHandles.m_bufferHandle);
        Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
        Obj inCBuffer = manager->GetCBuffer(in->m_dataHandles.m_cbufferHandle);
        if (!inBuffer || !outCBuffer || !inCBuffer)
        {
            return;
        }

        Send<void>(encoder, "setComputePipelineState:", pipeline);
        Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", inBuffer, 0ull, 1ull);
        MetalBindCBuffer(encoder, outCBuffer, 2ull);
        MetalBindCBuffer(encoder, inCBuffer, 3ull);
    }
    else
    {
        if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                          _out->GetDimsX(),
                                          _out->GetDimsY(),
                                          _out->GetDimsZ(),
                                          _out->GetOffset(),
                                          static_cast<uint32_t>(_out->GetID()))
            || !manager->UpdateCachedCBuffer(_L->m_dataHandles.m_cbufferHandle,
                                             _L->GetDimsX(),
                                             _L->GetDimsY(),
                                             _L->GetDimsZ(),
                                             _L->GetOffset(),
                                             static_cast<uint32_t>(_L->GetID()))
            || !manager->UpdateCachedCBuffer(_R->m_dataHandles.m_cbufferHandle,
                                             _R->GetDimsX(),
                                             _R->GetDimsY(),
                                             _R->GetDimsZ(),
                                             _R->GetOffset(),
                                             static_cast<uint32_t>(_R->GetID())))
        {
            return;
        }

        Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
        Obj leftCBuffer = manager->GetCBuffer(_L->m_dataHandles.m_cbufferHandle);
        Obj rightCBuffer = manager->GetCBuffer(_R->m_dataHandles.m_cbufferHandle);
        if (!outCBuffer || !leftCBuffer || !rightCBuffer)
        {
            return;
        }

        Send<void>(encoder, "setComputePipelineState:", pipeline);
        Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", leftBuffer, 0ull, 1ull);
        Send<void>(encoder, "setBuffer:offset:atIndex:", rightBuffer, 0ull, 2ull);
        MetalBindCBuffer(encoder, outCBuffer, 3ull);
        MetalBindCBuffer(encoder, leftCBuffer, 4ull);
        MetalBindCBuffer(encoder, rightCBuffer, 5ull);
    }

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}


template<TransposeMode Mode = TransposeMode::None, class T>
void Mul(MatrixMetal<T>* _out,            MatrixMetal<T>* _L,              MatrixMetal<T>* _R,
         uint32_t         blockSize = 32, uint32_t        numThreads = 12, ThreadPool*    pool = nullptr)
{
    (void)blockSize;
    (void)numThreads;
    (void)pool;

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

    if (!_out || !_L || !_R)
    {
        return;
    }

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _L->Unmap();
    _R->Unmap();

    constexpr bool transposeLeft = Mode == TransposeMode::Left || Mode == TransposeMode::Both;
    constexpr bool transposeRight = Mode == TransposeMode::Right || Mode == TransposeMode::Both;
    const uint32_t inner = transposeLeft ? _L->GetDimsY() : _L->GetDimsX();
    const bool tiled = _out->GetDimsX() >= 16u && _out->GetDimsY() >= 16u && inner >= 16u;
    Obj simdPipeline = tiled ? MetalMatMulSIMDPipeline(manager, _out->GetDimsY(),
        _out->GetDimsX(), inner) : nullptr;
    const bool simd = simdPipeline != nullptr;
    Obj pipeline = simd ? simdPipeline : manager->GetPipeline(tiled ? "mul_tiled" : shaderName);
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID()))
        || !manager->UpdateCachedCBuffer(_L->m_dataHandles.m_cbufferHandle,
                                         _L->GetDimsX(),
                                         _L->GetDimsY(),
                                         _L->GetDimsZ(),
                                         _L->GetOffset(),
                                         static_cast<uint32_t>(_L->GetID()))
        || !manager->UpdateCachedCBuffer(_R->m_dataHandles.m_cbufferHandle,
                                         _R->GetDimsX(),
                                         _R->GetDimsY(),
                                         _R->GetDimsZ(),
                                         _R->GetOffset(),
                                         static_cast<uint32_t>(_R->GetID())))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj leftBuffer = manager->GetBuffer(_L->m_dataHandles.m_bufferHandle);
    Obj rightBuffer = manager->GetBuffer(_R->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj leftCBuffer = manager->GetCBuffer(_L->m_dataHandles.m_cbufferHandle);
    Obj rightCBuffer = manager->GetCBuffer(_R->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !leftBuffer || !rightBuffer || !outCBuffer || !leftCBuffer || !rightCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", leftBuffer, 0ull, 1ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", rightBuffer, 0ull, 2ull);
    MetalBindCBuffer(encoder, outCBuffer, 3ull);
    MetalBindCBuffer(encoder, leftCBuffer, 4ull);
    MetalBindCBuffer(encoder, rightCBuffer, 5ull);
    if (tiled)
    {
        const uint32_t flags = (transposeLeft ? 1u : 0u) | (transposeRight ? 2u : 0u);
        Send<void>(encoder, "setBytes:length:atIndex:", &flags, sizeof(flags), 6ull);
    }

    const uint32_t elementCount = _out->GetElementCount();

    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    if (tiled)
    {
        const uint32_t tile = simd ? 32u : 16u;
        MTLSize groups = {(_out->GetDimsX() + tile - 1u) / tile,
                          (_out->GetDimsY() + tile - 1u) / tile, _out->GetDimsZ()};
        MTLSize tileThreads = simd ? MTLSize{128u, 1u, 1u} : MTLSize{16u, 16u, 1u};
        Send<void>(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", groups, tileThreads);
    }
    else
        Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();

}

template<TransposeMode Mode = TransposeMode::None, class T>
void MatMul_Strided(MatrixMetal<T>* _out,
                    MatrixMetal<T>* _L,
                    MatrixMetal<T>* _R,
                    uint32_t L_rowStride,
                    uint32_t L_colStride,
                    uint32_t R_rowStride,
                    uint32_t R_colStride,
                    uint32_t outRowStride,
                    uint32_t outColStride,
                    uint32_t blockSize = 32,
                    uint32_t numThreads = 12,
                    ThreadPool* pool = nullptr,
                    uint32_t L_batchStride = UINT32_MAX,
                    uint32_t R_batchStride = UINT32_MAX,
                    uint32_t outBatchStride = UINT32_MAX)
{
    (void)blockSize;
    (void)numThreads;
    (void)pool;

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

    if (!_out || !_L || !_R)
    {
        return;
    }

    MetalManager* manager = _out->m_gInstance ? _out->m_gInstance : MetalManager::Instance();
    if (!manager)
    {
        return;
    }

    manager->DumpInfoQueueErrors();
    _out->Unmap();
    _L->Unmap();
    _R->Unmap();

    constexpr bool transposeLeft = Mode == TransposeMode::Left || Mode == TransposeMode::Both;
    constexpr bool transposeRight = Mode == TransposeMode::Right || Mode == TransposeMode::Both;
    const uint32_t inner = transposeLeft ? _L->GetDimsY() : _L->GetDimsX();
    const bool tiled = _out->GetDimsX() >= 16u && _out->GetDimsY() >= 16u && inner >= 16u;
    Obj simdPipeline = tiled ? MetalMatMulSIMDPipeline(manager, _out->GetDimsY(),
        _out->GetDimsX(), inner) : nullptr;
    const bool simd = simdPipeline != nullptr;
    Obj pipeline = simd ? simdPipeline : manager->GetPipeline(tiled ? "mul_tiled" : shaderName);
    Obj queue = manager->GetQueue();
    if (!pipeline || !queue)
    {
        return;
    }

    if (!manager->UpdateCachedCBuffer(_out->m_dataHandles.m_cbufferHandle,
                                      _out->GetDimsX(),
                                      _out->GetDimsY(),
                                      _out->GetDimsZ(),
                                      _out->GetOffset(),
                                      static_cast<uint32_t>(_out->GetID()))
        || !manager->UpdateCachedCBuffer(_L->m_dataHandles.m_cbufferHandle,
                                         _L->GetDimsX(),
                                         _L->GetDimsY(),
                                         _L->GetDimsZ(),
                                         _L->GetOffset(),
                                         static_cast<uint32_t>(_L->GetID()))
        || !manager->UpdateCachedCBuffer(_R->m_dataHandles.m_cbufferHandle,
                                         _R->GetDimsX(),
                                         _R->GetDimsY(),
                                         _R->GetDimsZ(),
                                         _R->GetOffset(),
                                         static_cast<uint32_t>(_R->GetID())))
    {
        return;
    }

    const uint32_t outParams[3] = { outRowStride, outColStride,
        outBatchStride == UINT32_MAX ? (outRowStride ? outRowStride : _out->GetDimsX()) * _out->GetDimsY() : outBatchStride };
    const uint32_t leftParams[3] = { L_rowStride, L_colStride,
        L_batchStride == UINT32_MAX ? (L_rowStride ? L_rowStride : _L->GetDimsX()) * _L->GetDimsY() : L_batchStride };
    const uint32_t rightParams[3] = { R_rowStride, R_colStride,
        R_batchStride == UINT32_MAX ? (R_rowStride ? R_rowStride : _R->GetDimsX()) * _R->GetDimsY() : R_batchStride };
    if (!manager->SetCBufferOptionalParams(_out->m_dataHandles.m_cbufferHandle, outParams, 3u)
        || !manager->SetCBufferOptionalParams(_L->m_dataHandles.m_cbufferHandle, leftParams, 3u)
        || !manager->SetCBufferOptionalParams(_R->m_dataHandles.m_cbufferHandle, rightParams, 3u))
    {
        return;
    }

    Obj outBuffer = manager->GetBuffer(_out->m_dataHandles.m_bufferHandle);
    Obj leftBuffer = manager->GetBuffer(_L->m_dataHandles.m_bufferHandle);
    Obj rightBuffer = manager->GetBuffer(_R->m_dataHandles.m_bufferHandle);
    Obj outCBuffer = manager->GetCBuffer(_out->m_dataHandles.m_cbufferHandle);
    Obj leftCBuffer = manager->GetCBuffer(_L->m_dataHandles.m_cbufferHandle);
    Obj rightCBuffer = manager->GetCBuffer(_R->m_dataHandles.m_cbufferHandle);
    if (!outBuffer || !leftBuffer || !rightBuffer || !outCBuffer || !leftCBuffer || !rightCBuffer)
    {
        return;
    }

    Obj commandBuffer = manager->GetCommandBuffer();
    Obj encoder = manager->GetEncoder();
    if (!commandBuffer || !encoder)
    {
        return;
    }

    Send<void>(encoder, "setComputePipelineState:", pipeline);
    Send<void>(encoder, "setBuffer:offset:atIndex:", outBuffer, 0ull, 0ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", leftBuffer, 0ull, 1ull);
    Send<void>(encoder, "setBuffer:offset:atIndex:", rightBuffer, 0ull, 2ull);
    MetalBindCBuffer(encoder, outCBuffer, 3ull);
    MetalBindCBuffer(encoder, leftCBuffer, 4ull);
    MetalBindCBuffer(encoder, rightCBuffer, 5ull);
    if (tiled)
    {
        const uint32_t flags = (transposeLeft ? 1u : 0u) | (transposeRight ? 2u : 0u) | 4u;
        Send<void>(encoder, "setBytes:length:atIndex:", &flags, sizeof(flags), 6ull);
    }

    const uint32_t elementCount = _out->GetElementCount();
    MTLSize threads = {elementCount, 1, 1};
    MTLSize threadsPerThreadgroup = {256u, 1, 1};
    if (tiled)
    {
        const uint32_t tile = simd ? 32u : 16u;
        MTLSize groups = {(_out->GetDimsX() + tile - 1u) / tile,
                          (_out->GetDimsY() + tile - 1u) / tile, _out->GetDimsZ()};
        MTLSize tileThreads = simd ? MTLSize{128u, 1u, 1u} : MTLSize{16u, 16u, 1u};
        Send<void>(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", groups, tileThreads);
    }
    else
        Send<void>(encoder, "dispatchThreads:threadsPerThreadgroup:", threads, threadsPerThreadgroup);

    manager->WorkSubmitted();
    manager->DumpInfoQueueErrors();
}
 
// All gradient reads, reductions, scale generation and scaling remain queued.
template<class T>
void ClipGradientNorm(const std::vector<MatrixMetal<T>*>& gradients, float maximumNorm,
                      GradientClipWorkspace<T,MatrixMetal<T>>& workspace)
{
    if (!std::isfinite(maximumNorm) || maximumNorm < 0.0f)
        throw std::invalid_argument("Gradient clip norm must be finite and nonnegative.");
    if (maximumNorm == 0.0f) return;
    const uint32_t partialCount = workspace.Prepare(gradients);
    if (partialCount == 0u) return;
    auto* manager = workspace.partials->m_gInstance;
    Obj reducePipeline = manager->GetPipeline("gradient_norm_partials");
    Obj scalePipeline = manager->GetPipeline("gradient_norm_scale");
    if (!reducePipeline || !scalePipeline)
        throw std::runtime_error("Missing GPU gradient norm pipeline.");
    workspace.partials->Unmap(); workspace.scale->Unmap();
    if (!MetalSyncCBuffer(workspace.partials.get()) || !MetalSyncCBuffer(workspace.scale.get()))
        throw std::runtime_error("Cannot bind gradient norm scratch buffers.");
    Obj partialBuffer = manager->GetBuffer(workspace.partials->m_dataHandles.m_bufferHandle);
    Obj partialCB = manager->GetCBuffer(workspace.partials->m_dataHandles.m_cbufferHandle);
    uint32_t destination = 0u;
    for (auto* gradient : gradients)
    {
        if (gradient->GetElementCount() == 0u) continue;
        gradient->Unmap();
        if (!MetalSyncCBuffer(gradient)) throw std::runtime_error("Cannot bind gradient buffer.");
        Obj encoder = manager->GetEncoder();
        Send<void>(encoder,"setComputePipelineState:",reducePipeline);
        Send<void>(encoder,"setBuffer:offset:atIndex:",partialBuffer,0ull,0ull);
        Send<void>(encoder,"setBuffer:offset:atIndex:",manager->GetBuffer(gradient->m_dataHandles.m_bufferHandle),0ull,1ull);
        MetalBindCBuffer(encoder,partialCB,2ull);
        MetalBindCBuffer(encoder,manager->GetCBuffer(gradient->m_dataHandles.m_cbufferHandle),3ull);
        Send<void>(encoder,"setBytes:length:atIndex:",&destination,sizeof(destination),4ull);
        const uint32_t groups = uint32_t((uint64_t(gradient->GetElementCount())+1023u)/1024u);
        Send<void>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{groups,1u,1u},MTLSize{256u,1u,1u});
        manager->WorkSubmitted();
        destination += groups;
    }
    const uint32_t params[] = {MetalPackFloatBits(maximumNorm),partialCount};
    manager->SetCBufferOptionalParams(workspace.scale->m_dataHandles.m_cbufferHandle,params,2u);
    Obj encoder = manager->GetEncoder();
    Send<void>(encoder,"setComputePipelineState:",scalePipeline);
    Send<void>(encoder,"setBuffer:offset:atIndex:",manager->GetBuffer(workspace.scale->m_dataHandles.m_bufferHandle),0ull,0ull);
    Send<void>(encoder,"setBuffer:offset:atIndex:",partialBuffer,0ull,1ull);
    MetalBindCBuffer(encoder,manager->GetCBuffer(workspace.scale->m_dataHandles.m_cbufferHandle),2ull);
    MetalBindCBuffer(encoder,partialCB,3ull);
    Send<void>(encoder,"dispatchThreadgroups:threadsPerThreadgroup:",MTLSize{1u,1u,1u},MTLSize{256u,1u,1u});
    manager->WorkSubmitted();
    for (auto* gradient : gradients)
        if (gradient->GetElementCount() != 0u) Scale(gradient,gradient,workspace.scale.get());
}

#endif
