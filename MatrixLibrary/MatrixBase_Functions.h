#pragma once

#include <MatrixLibrary/MatrixBase.h>
#include <MatrixLibrary/GradientClipWorkspace.h>
#include <ToolsLibrary/ThreadPool.h>
#include <ToolsLibrary/Tools.h>

#include <latch>
#include <cassert>
#include <cstring>
#include <random>
#include <vector>
#include <cmath>

// Functions implemented here
// Add
// Add (Broadcast)
// Sub
// Mul
// Mul (Per Element)
// Map Functions
// Scale
// Fill
// Min/Max
// Sum Columns

template<class T>
T MeanSquaredErrorLoss(MatrixBase<T>* target, MatrixBase<T>* prediction)
{
    assert(target);
    assert(prediction);
    assert(target->GetElementCount() == prediction->GetElementCount());

    const size_t count = target->GetElementCount();
    if (count == 0u)
    {
        return T(0);
    }

    const T* targetPtr = target->DataRead();
    const T* predictionPtr = prediction->DataRead();
    T sum = T(0);
    for (size_t i = 0; i < count; ++i)
    {
        const T d = targetPtr[i] - predictionPtr[i];
        sum += d * d;
    }
    return sum / static_cast<T>(count);
}

template<class T>
void Softmax(MatrixBase<T>* out, MatrixBase<T>* in)
{
    assert(out->GetWidth() == in->GetWidth() && out->GetHeight() == in->GetHeight());
    const uint32_t rows = in->GetHeight() * in->GetPlaneCount();
    const uint32_t cols = in->GetWidth();
    const T* inPtr = in->DataRead();
    T* outPtr = out->DataWrite();

    for (uint32_t row = 0; row < rows; ++row)
    {
        const uint32_t rowOffset = row * cols;
        T maxVal = inPtr[rowOffset];
        for (uint32_t col = 1; col < cols; ++col)
        {
            maxVal = std::max(maxVal, inPtr[rowOffset + col]);
        }

        T sum = T(0);
        for (uint32_t col = 0; col < cols; ++col)
        {
            const T v = std::exp(inPtr[rowOffset + col] - maxVal);
            outPtr[rowOffset + col] = v;
            sum += v;
        }

        for (uint32_t col = 0; col < cols; ++col)
        {
            outPtr[rowOffset + col] /= sum;
        }
    }
}

template<class T>
void SoftmaxBackwards(MatrixBase<T>* out, MatrixBase<T>* softmaxOut, MatrixBase<T>* gradIn)
{
    assert(out);
    assert(softmaxOut);
    assert(gradIn);
    assert(out->GetWidth() == softmaxOut->GetWidth() && out->GetHeight() == softmaxOut->GetHeight());
    assert(gradIn->GetWidth() == softmaxOut->GetWidth() && gradIn->GetHeight() == softmaxOut->GetHeight());

    const uint32_t rows = softmaxOut->GetHeight() * softmaxOut->GetPlaneCount();
    const uint32_t cols = softmaxOut->GetWidth();
    const T* softPtr = softmaxOut->DataRead();
    const T* gradPtr = gradIn->DataRead();
    T* outPtr = out->DataWrite();

    for (uint32_t row = 0; row < rows; ++row)
    {
        const uint32_t rowOffset = row * cols;
        T dot = T(0);
        for (uint32_t col = 0; col < cols; ++col)
        {
            dot += gradPtr[rowOffset + col] * softPtr[rowOffset + col];
        }
        for (uint32_t col = 0; col < cols; ++col)
        {
            outPtr[rowOffset + col] = softPtr[rowOffset + col] * (gradPtr[rowOffset + col] - dot);
        }
    }
}

// Row-major square attention planes; Z contains heads and/or batches.
// Probs are pre-dropout softmax outputs; grad has already passed dropout backward.
template<class T>
void AttentionScoresBackwards(MatrixBase<T>* out, MatrixBase<T>* probs,
                              MatrixBase<T>* grad, MatrixBase<T>* scale)
{
    assert(out && probs && grad && scale);
    assert(out->GetWidth() == out->GetHeight());
    assert(out->GetWidth() == probs->GetWidth() && out->GetHeight() == probs->GetHeight()
           && out->GetPlaneCount() == probs->GetPlaneCount());
    assert(out->GetWidth() == grad->GetWidth() && out->GetHeight() == grad->GetHeight()
           && out->GetPlaneCount() == grad->GetPlaneCount());
    assert(scale->GetElementCount() == 1u);
    const uint32_t width = out->GetWidth();
    const uint32_t rows = out->GetHeight() * out->GetPlaneCount();
    const T* p = probs->DataRead();
    const T* g = grad->DataRead();
    const T factor = scale->DataRead()[0];
    T* result = out->DataWrite();
    for (uint32_t row = 0; row < rows; ++row)
    {
        const uint32_t offset = row * width;
        const uint32_t query = row % width;
        T dot = T(0);
        for (uint32_t col = 0; col < width; ++col)
            dot += p[offset + col] * g[offset + col];
        for (uint32_t col = 0; col <= query; ++col)
            result[offset + col] = (p[offset + col] * (g[offset + col] - dot)) * factor;
        std::fill(result + offset + query + 1u, result + offset + width, T(0));
    }
}

enum class TriangleDirection { Lower, Upper };
template<class T>
void TriangleMatrix(MatrixBase<T>* _out, TriangleDirection _dir, T trueVal, T falseVal)
{
    assert(_out);

    const uint32_t rows = _out->GetHeight() * _out->GetPlaneCount();
    const uint32_t cols = _out->GetWidth();
    const T* inPtr = _out->DataRead();
    T* outPtr = _out->DataWrite();

    if (_dir == TriangleDirection::Lower)
    {
        for (uint32_t row = 0; row < rows; ++row)
        {
            const uint32_t rowOffset = row * cols;
            const uint32_t keep = row % _out->GetHeight() + 1 < cols ? row % _out->GetHeight() + 1 : cols;
            for (uint32_t col = 0; col < keep; ++col)
            {
                outPtr[rowOffset + col] = trueVal;
            }
            for (uint32_t col = keep; col < cols; ++col)
            {
                outPtr[rowOffset + col] = falseVal;
            }
        }
        return;
    }

    for (uint32_t row = 0; row < rows; ++row)
    {
        const uint32_t rowOffset = row * cols;
        const uint32_t start = row % _out->GetHeight() < cols ? row % _out->GetHeight() : cols;
        for (uint32_t col = 0; col < start; ++col)
        {
            outPtr[rowOffset + col] = falseVal;
        }
        for (uint32_t col = start; col < cols; ++col)
        {
            outPtr[rowOffset + col] = trueVal;
        }
    }
}

template<class T>
void Copy(MatrixBase<T>* _out, MatrixBase<T>* _in)
{
    assert(_out->GetElementCount() == _in->GetElementCount());
    std::memcpy(_out->DataWrite(), _in->DataRead(), sizeof(T) * _in->GetElementCount());
}

template<class T>
void CopyRange(MatrixBase<T>* _out,
               MatrixBase<T>* _in,
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
    std::memcpy(_out->DataWrite() + outOffset,
                _in->DataRead() + inOffset,
                sizeof(T) * count);
}

template<class Mat>
void Concat(Mat* out,Mat* left,Mat* right,MatrixAxis axis)
{
    if(!out || !left || !right || out->GetShape()!=left->GetShape().Concatenated(right->GetShape(),axis))
        throw std::invalid_argument("Concatenation output shape is incorrect.");
    uint32_t leftChunk=left->GetWidth(),rightChunk=right->GetWidth();
    uint32_t outer=left->GetHeight()*left->GetPlaneCount();
    if(axis!=MatrixAxis::Width) {
        leftChunk*=left->GetHeight();rightChunk*=right->GetHeight();outer=left->GetPlaneCount();
        if(axis!=MatrixAxis::Height) {
            leftChunk*=left->GetChannelCount();rightChunk*=right->GetChannelCount();outer=left->GetBatchCount();
            if(axis==MatrixAxis::Batch) {leftChunk*=left->GetBatchCount();rightChunk*=right->GetBatchCount();outer=1u;}
        }
    }
    for(uint32_t index=0;index<outer;++index) {
        CopyRange(out,left,index*(leftChunk+rightChunk),index*leftChunk,leftChunk);
        CopyRange(out,right,index*(leftChunk+rightChunk)+leftChunk,index*rightChunk,rightChunk);
    }
}

template<class T>
void ConcatZ(MatrixBase<T>* _out, MatrixBase<T>* _A, MatrixBase<T>* _B)
{
    assert(_out);
    assert(_A);
    assert(_B);
    assert(_A->GetWidth() == _B->GetWidth());
    assert(_A->GetHeight() == _B->GetHeight());
    assert(_out->GetWidth() == _A->GetWidth());
    assert(_out->GetHeight() == _A->GetHeight());
    assert(_out->GetPlaneCount() == _A->GetPlaneCount() + _B->GetPlaneCount());

    const uint32_t countA = _A->GetElementCount();
    const uint32_t countB = _B->GetElementCount();

    std::memcpy(_out->DataWrite(), _A->DataRead(), sizeof(T) * countA);
    std::memcpy(_out->DataWrite() + countA, _B->DataRead(), sizeof(T) * countB);
}

template<class T>
void GatherRows(MatrixBase<T>* _out, MatrixBase<T>* _src, MatrixBase<T>* _indices)
{
    assert(_out);
    assert(_src);
    assert(_indices);
    assert(_indices->GetWidth() == 1u);

    const uint32_t rowWidth = _src->GetWidth();
    const uint32_t outRows = _out->GetHeight();
    const uint32_t outBatch = _out->GetPlaneCount();
    const uint32_t srcRows = _src->GetHeight();
    const uint32_t srcBatch = _src->GetPlaneCount();

    assert(_out->GetWidth() == rowWidth);
    assert(_indices->GetHeight() == outRows);
    assert(_indices->GetPlaneCount() == outBatch);
    assert(srcBatch == 1u || srcBatch == outBatch);

    const uint32_t outPlane = rowWidth * outRows;
    const uint32_t srcPlane = rowWidth * srcRows;
    const T* srcPtr = _src->DataRead();
    const T* idxPtr = _indices->DataRead();
    T* outPtr = _out->DataWrite();

    for (uint32_t z = 0; z < outBatch; ++z)
    {
        const uint32_t srcZ = (srcBatch == 1u) ? 0u : z;
        const uint32_t outBase = z * outPlane;
        const uint32_t srcBase = srcZ * srcPlane;
        const uint32_t idxBase = z * outRows;
        for (uint32_t row = 0; row < outRows; ++row)
        {
            const uint32_t srcRow = static_cast<uint32_t>(idxPtr[idxBase + row]);
            assert(srcRow < srcRows);
            std::memcpy(outPtr + outBase + row * rowWidth,
                        srcPtr + srcBase + srcRow * rowWidth,
                        sizeof(T) * rowWidth);
        }
    }
}

template<class T>
void ScatterRows(MatrixBase<T>* _out, MatrixBase<T>* _src, MatrixBase<T>* _indices)
{
    assert(_out);
    assert(_src);
    assert(_indices);
    assert(_indices->GetWidth() == 1u);
    assert(_src->GetWidth() == _out->GetWidth());
    assert(_src->GetHeight() == _indices->GetHeight());
    assert(_src->GetPlaneCount() == _indices->GetPlaneCount());
    assert(_out->GetPlaneCount() == 1u || _out->GetPlaneCount() == _src->GetPlaneCount());

    const uint32_t rowWidth = _src->GetWidth();
    const uint32_t srcRows = _src->GetHeight();
    const uint32_t srcBatch = _src->GetPlaneCount();
    const uint32_t outRows = _out->GetHeight();
    const uint32_t outBatch = _out->GetPlaneCount();

    const uint32_t srcPlane = rowWidth * srcRows;
    const uint32_t outPlane = rowWidth * outRows;
    const T* srcPtr = _src->DataRead();
    const T* idxPtr = _indices->DataRead();
    T* outPtr = _out->DataWrite();

    for (uint32_t z = 0; z < srcBatch; ++z)
    {
        const uint32_t outZ = (outBatch == 1u) ? 0u : z;
        const uint32_t srcBase = z * srcPlane;
        const uint32_t outBase = outZ * outPlane;
        const uint32_t idxBase = z * srcRows;
        for (uint32_t row = 0; row < srcRows; ++row)
        {
            const uint32_t dstRow = static_cast<uint32_t>(idxPtr[idxBase + row]);
            assert(dstRow < outRows);
            std::memcpy(outPtr + outBase + dstRow * rowWidth,
                        srcPtr + srcBase + row * rowWidth,
                        sizeof(T) * rowWidth);
        }
    }
}

template<class T>
void ScatterAddRows(MatrixBase<T>* _out, MatrixBase<T>* _src, MatrixBase<T>* _indices)
{
    assert(_out);
    assert(_src);
    assert(_indices);
    assert(_indices->GetWidth() == 1u);
    assert(_src->GetWidth() == _out->GetWidth());
    assert(_src->GetHeight() == _indices->GetHeight());
    assert(_src->GetPlaneCount() == _indices->GetPlaneCount());
    assert(_out->GetPlaneCount() == 1u || _out->GetPlaneCount() == _src->GetPlaneCount());

    const uint32_t rowWidth = _src->GetWidth();
    const uint32_t srcRows = _src->GetHeight();
    const uint32_t srcBatch = _src->GetPlaneCount();
    const uint32_t outRows = _out->GetHeight();
    const uint32_t outBatch = _out->GetPlaneCount();

    const uint32_t srcPlane = rowWidth * srcRows;
    const uint32_t outPlane = rowWidth * outRows;
    const T* srcPtr = _src->DataRead();
    const T* idxPtr = _indices->DataRead();
    T* outPtr = _out->DataWrite();

    for (uint32_t z = 0; z < srcBatch; ++z)
    {
        const uint32_t outZ = (outBatch == 1u) ? 0u : z;
        const uint32_t srcBase = z * srcPlane;
        const uint32_t outBase = outZ * outPlane;
        const uint32_t idxBase = z * srcRows;
        for (uint32_t row = 0; row < srcRows; ++row)
        {
            const uint32_t dstRow = static_cast<uint32_t>(idxPtr[idxBase + row]);
            assert(dstRow < outRows);
            const T* srcRowPtr = srcPtr + srcBase + row * rowWidth;
            T* dstRowPtr = outPtr + outBase + dstRow * rowWidth;
            for (uint32_t x = 0; x < rowWidth; ++x)
            {
                dstRowPtr[x] += srcRowPtr[x];
            }
        }
    }
}

template<class T>
void SplitQKV(MatrixBase<T>* _q,
              MatrixBase<T>* _k,
              MatrixBase<T>* _v,
              MatrixBase<T>* _packed)
{
    assert(_q);
    assert(_k);
    assert(_v);
    assert(_packed);

    const uint32_t dC = _q->GetWidth();
    const uint32_t rows = _q->GetHeight() * _q->GetPlaneCount();
    const uint32_t qCount = _q->GetElementCount();
    if (qCount == 0u)
    {
        return;
    }

    assert(_k->GetElementCount() == qCount);
    assert(_v->GetElementCount() == qCount);
    assert(_packed->GetElementCount() == qCount * 3u);
    assert(_packed->GetWidth() == dC * 3u);
    assert((_packed->GetHeight() * _packed->GetPlaneCount()) == rows);

    const uint32_t qkvCols = dC * 3u;
    T* qPtr = _q->DataWrite();
    T* kPtr = _k->DataWrite();
    T* vPtr = _v->DataWrite();
    const T* packedPtr = _packed->DataRead();
    for (uint32_t row = 0; row < rows; ++row)
    {
        const uint32_t packedOffset = row * qkvCols;
        const uint32_t outOffset = row * dC;
        std::memcpy(qPtr + outOffset, packedPtr + packedOffset, sizeof(T) * dC);
        std::memcpy(kPtr + outOffset, packedPtr + packedOffset + dC, sizeof(T) * dC);
        std::memcpy(vPtr + outOffset, packedPtr + packedOffset + 2u * dC, sizeof(T) * dC);
    }
}

template<class T>
void MergeQKV(MatrixBase<T>* _packedQKV,
              MatrixBase<T>* _qHeads,
              MatrixBase<T>* _kHeads,
              MatrixBase<T>* _vHeads)
{
    assert(_packedQKV);
    assert(_qHeads);
    assert(_kHeads);
    assert(_vHeads);

    const uint32_t headDim = _qHeads->GetWidth();
    const uint32_t tokens = _qHeads->GetHeight();
    const uint32_t nHead = _qHeads->GetChannelCount();
    const uint32_t dC = headDim * nHead;
    const uint32_t qCount = _qHeads->GetElementCount();
    if (qCount == 0u)
    {
        return;
    }

    assert(_kHeads->GetWidth() == headDim);
    assert(_kHeads->GetHeight() == tokens);
    assert(_kHeads->GetChannelCount() == nHead);
    assert(_vHeads->GetWidth() == headDim);
    assert(_vHeads->GetHeight() == tokens);
    assert(_vHeads->GetChannelCount() == nHead);
    assert(_kHeads->GetElementCount() == qCount);
    assert(_vHeads->GetElementCount() == qCount);

    assert(_packedQKV->GetWidth() == 3u * dC);
    assert(_packedQKV->GetHeight() == tokens);
    assert(_packedQKV->GetChannelCount() == 1u && _packedQKV->GetBatchCount() == _qHeads->GetBatchCount());
    assert(_packedQKV->GetElementCount() == qCount * 3u);

    const uint32_t qkvCols = 3u * dC;
    const uint32_t headPlane = headDim * tokens;

    const T* qPtr = _qHeads->DataRead();
    const T* kPtr = _kHeads->DataRead();
    const T* vPtr = _vHeads->DataRead();
    T* packedPtr = _packedQKV->DataWrite();

    for (uint32_t batch = 0; batch < _qHeads->GetBatchCount(); ++batch)
    for (uint32_t row = 0; row < tokens; ++row)
    {
        const uint32_t rowPackedBase = (batch * tokens + row) * qkvCols;
        const uint32_t rowHeadOffset = row * headDim;
        for (uint32_t head = 0; head < nHead; ++head)
        {
            const uint32_t srcOffset = (batch * nHead + head) * headPlane + rowHeadOffset;
            const uint32_t dstOffset = rowPackedBase + head * headDim;
            std::memcpy(packedPtr + dstOffset,
                        qPtr + srcOffset,
                        sizeof(T) * headDim);
            std::memcpy(packedPtr + dstOffset + dC,
                        kPtr + srcOffset,
                        sizeof(T) * headDim);
            std::memcpy(packedPtr + dstOffset + 2u * dC,
                        vPtr + srcOffset,
                        sizeof(T) * headDim);
        }
    }
}

template<class T>
void TransposeMat(MatrixBase<T>* _out, MatrixBase<T>* _in)
{
    assert(_out);
    assert(_in);
    assert(_out != _in);
    assert(_out->GetWidth() == _in->GetHeight());
    assert(_out->GetHeight() == _in->GetWidth());
    assert(_out->GetPlaneCount() == _in->GetPlaneCount());

    const uint32_t inX = _in->GetWidth();
    const uint32_t inY = _in->GetHeight();
    const uint32_t inZ = _in->GetPlaneCount();
    const uint32_t outX = _out->GetWidth();
    const uint32_t outY = _out->GetHeight();

    const T* inPtr = _in->DataRead();
    T* outPtr = _out->DataWrite();
    const uint32_t inPlane = inX * inY;
    const uint32_t outPlane = outX * outY;

    for (uint32_t z = 0; z < inZ; ++z)
    {
        const uint32_t inBase = z * inPlane;
        const uint32_t outBase = z * outPlane;
        for (uint32_t y = 0; y < outY; ++y)
        {
            const uint32_t outRow = outBase + (y * outX);
            for (uint32_t x = 0; x < outX; ++x)
            {
                const uint32_t inIdx = inBase + (x * inX) + y;
                outPtr[outRow + x] = inPtr[inIdx];
            }
        }
    }
}

template<class T>
void BroadcastBinary(MatrixBase<T>* out, MatrixBase<T>* left, MatrixBase<T>* right, uint32_t operation)
{
    if (!out || !left || !right || operation>2u
        || out->GetShape()!=left->GetShape().BroadcastWith(right->GetShape()))
        throw std::invalid_argument("Broadcast operation output shape is incorrect.");
    const auto& shape=out->GetShape();const auto& l=left->GetShape();const auto& r=right->GetShape();
    const T* a=left->DataRead();const T* b=right->DataRead();T* result=out->DataWrite();
    for(uint32_t batch=0;batch<shape.batch;++batch) for(uint32_t c=0;c<shape.channels;++c)
    for(uint32_t y=0;y<shape.height;++y) for(uint32_t x=0;x<shape.width;++x) {
        const T lhs=a[l.Index(l.batch==1u?0u:batch,l.channels==1u?0u:c,l.width==1u?0u:x,l.height==1u?0u:y)];
        const T rhs=b[r.Index(r.batch==1u?0u:batch,r.channels==1u?0u:c,r.width==1u?0u:x,r.height==1u?0u:y)];
        result[shape.Index(batch,c,x,y)]=operation==0u?lhs+rhs:operation==1u?lhs-rhs:lhs*rhs;
    }
}

template<class T>
void Add(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{
    if (_out->GetShape()!=_L->GetShape() || _out->GetShape()!=_R->GetShape())
    { BroadcastBinary(_out,_L,_R,0u);return; }
    assert(_L->GetElementCount() == _R->GetElementCount());
    for (uint32_t i = 0; i < _R->GetElementCount(); i++)
    {
        const T lhs = _L->DataRead()[i];
        const T rhs = _R->DataRead()[i];
        _out->DataWrite()[i] = lhs + rhs;
    }
}

template<class T>
void LayerNormOp(MatrixBase<T>* _out,
                 MatrixBase<T>* _in,
                 MatrixBase<T>* _gamma,
                 MatrixBase<T>* _beta = nullptr,
                 MatrixBase<T>* _xHat = nullptr,
                 T _eps = static_cast<T>(1e-5),
                 MatrixBase<T>* _invStd = nullptr)
{
    assert(_out);
    assert(_in);
    assert(_gamma);
    if (_invStd)
    {
        assert(_xHat);
        assert(_invStd->GetWidth() == 1u);
        assert(_invStd->GetHeight() == _in->GetHeight());
        assert(_invStd->GetPlaneCount() == _in->GetPlaneCount());
    }

    assert(_out->GetWidth() == _in->GetWidth());
    assert(_out->GetHeight() == _in->GetHeight());
    assert(_out->GetPlaneCount() == _in->GetPlaneCount());
    assert(_gamma->GetWidth() == _in->GetWidth());
    assert(_gamma->GetElementCount() >= _in->GetWidth());
    if (_beta)
    {
        assert(_beta->GetWidth() == _in->GetWidth());
        assert(_beta->GetElementCount() >= _in->GetWidth());
    }
    if (_xHat)
    {
        assert(_xHat->GetWidth() == _in->GetWidth());
        assert(_xHat->GetHeight() == _in->GetHeight());
        assert(_xHat->GetPlaneCount() == _in->GetPlaneCount());
    }

    const uint32_t cols = _in->GetWidth();
    const uint32_t rows = _in->GetHeight() * _in->GetPlaneCount();
    const T invCols = static_cast<T>(1) / static_cast<T>(cols);

    const T* inPtr = _in->DataRead();
    const T* gPtr = _gamma->DataRead();
    const T* bPtr = _beta ? _beta->DataRead() : nullptr;
    T* outPtr = _out->DataWrite();
    T* xHatPtr = _xHat ? _xHat->DataWrite() : nullptr;

    T* invStdPtr = _invStd ? _invStd->DataWrite() : nullptr;

    for (uint32_t row = 0; row < rows; ++row)
    {
        const uint32_t rowOffset = row * cols;
        T mean = T(0);
        for (uint32_t col = 0; col < cols; ++col)
        {
            mean += inPtr[rowOffset + col];
        }
        mean *= invCols;

        T var = T(0);
        for (uint32_t col = 0; col < cols; ++col)
        {
            const T diff = inPtr[rowOffset + col] - mean;
            var += diff * diff;
        }
        var *= invCols;
        const T invStd = static_cast<T>(1) / std::sqrt(var + _eps);
        if (invStdPtr) invStdPtr[row] = invStd;

        for (uint32_t col = 0; col < cols; ++col)
        {
            const T normalized = (inPtr[rowOffset + col] - mean) * invStd;
            if (xHatPtr)
            {
                xHatPtr[rowOffset + col] = normalized;
            }
            const T bias = bPtr ? bPtr[col] : T(0);
            outPtr[rowOffset + col] = normalized * gPtr[col] + bias;
        }
    }
}

template<class T>
void LayerNormBackwardsOp(MatrixBase<T>* _dX,
                          MatrixBase<T>* _dY,
                          MatrixBase<T>* _in,
                          MatrixBase<T>* _gamma,
                          MatrixBase<T>* _dGamma,
                          MatrixBase<T>* _dBeta = nullptr,
                          MatrixBase<T>* _xHat = nullptr,
                          T _eps = static_cast<T>(1e-5),
                          MatrixBase<T>* _invStd = nullptr)
{
    assert(_dX);
    assert(_dY);
    assert(_in);
    assert(_gamma);
    if (_invStd)
    {
        assert(_xHat);
        assert(_invStd->GetWidth() == 1u);
        assert(_invStd->GetHeight() == _in->GetHeight());
        assert(_invStd->GetPlaneCount() == _in->GetPlaneCount());
    }

    assert(_dGamma);
    assert(_dX->GetWidth() == _in->GetWidth());
    assert(_dX->GetHeight() == _in->GetHeight());
    assert(_dX->GetPlaneCount() == _in->GetPlaneCount());
    assert(_dY->GetWidth() == _in->GetWidth());
    assert(_dY->GetHeight() == _in->GetHeight());
    assert(_dY->GetPlaneCount() == _in->GetPlaneCount());
    assert(_gamma->GetWidth() == _in->GetWidth());
    assert(_gamma->GetElementCount() >= _in->GetWidth());
    assert(_dGamma->GetWidth() == _in->GetWidth());
    assert(_dGamma->GetElementCount() >= _in->GetWidth());
    if (_dBeta)
    {
        assert(_dBeta->GetWidth() == _in->GetWidth());
        assert(_dBeta->GetElementCount() >= _in->GetWidth());
    }
    if (_xHat)
    {
        assert(_xHat->GetWidth() == _in->GetWidth());
        assert(_xHat->GetHeight() == _in->GetHeight());
        assert(_xHat->GetPlaneCount() == _in->GetPlaneCount());
    }

    const uint32_t rowsTotal = _in->GetHeight() * _in->GetPlaneCount();
    const uint32_t cols = _in->GetWidth();
    const T invCols = static_cast<T>(1) / static_cast<T>(cols);

    const T* inPtr = (_invStd && _xHat) ? nullptr : _in->DataRead();
    const T* dYPtr = _dY->DataRead();
    const T* gPtr = _gamma->DataRead();
    T* dXPtr = _dX->DataWrite();

    // dBeta = sum(dY) over rows; dGamma = sum(dY * Xhat) over rows.
    {
        T* gGradPtr = _dGamma->DataWrite();
        std::memset(gGradPtr, 0, sizeof(T) * cols);
        T* bGradPtr = _dBeta ? _dBeta->DataWrite() : nullptr;
        if (bGradPtr)
        {
            std::memset(bGradPtr, 0, sizeof(T) * cols);
        }

        if (_xHat)
        {
            const T* xHatAll = _xHat->DataRead();
            for (uint32_t row = 0; row < rowsTotal; ++row)
            {
                const uint32_t rowOffset = row * cols;
                for (uint32_t col = 0; col < cols; ++col)
                {
                    const T dy = dYPtr[rowOffset + col];
                    if (bGradPtr)
                    {
                        bGradPtr[col] += dy;
                    }
                    gGradPtr[col] += dy * xHatAll[rowOffset + col];
                }
            }
        }
        else
        {
            // Compute xHat on the fly if not provided.
            for (uint32_t row = 0; row < rowsTotal; ++row)
            {
                const uint32_t rowOffset = row * cols;
                T mean = T(0);
                for (uint32_t col = 0; col < cols; ++col)
                {
                    mean += inPtr[rowOffset + col];
                }
                mean *= invCols;

                T var = T(0);
                for (uint32_t col = 0; col < cols; ++col)
                {
                    const T diff = inPtr[rowOffset + col] - mean;
                    var += diff * diff;
                }
                var *= invCols;
                const T invStd = static_cast<T>(1) / std::sqrt(var + _eps);

                for (uint32_t col = 0; col < cols; ++col)
                {
                    const T dy = dYPtr[rowOffset + col];
                    const T xHat = (inPtr[rowOffset + col] - mean) * invStd;
                    if (bGradPtr)
                    {
                        bGradPtr[col] += dy;
                    }
                    gGradPtr[col] += dy * xHat;
                }
            }
        }
    }

    if (_invStd && _xHat)
    {
        const T* invStdPtr = _invStd->DataRead();
        const T* hatPtr = _xHat->DataRead();
        for (uint32_t row = 0; row < rowsTotal; ++row)
        {
            const uint32_t offset = row * cols;
            T sumGrad = T(0), sumGradHat = T(0);
            for (uint32_t col = 0; col < cols; ++col)
            {
                const T grad = dYPtr[offset + col] * gPtr[col];
                sumGrad += grad;
                sumGradHat += grad * hatPtr[offset + col];
            }
            for (uint32_t col = 0; col < cols; ++col)
                dXPtr[offset + col] = invStdPtr[row] * (dYPtr[offset + col] * gPtr[col]
                    - sumGrad * invCols - hatPtr[offset + col] * sumGradHat * invCols);
        }
        return;
    }

    for (uint32_t row = 0; row < rowsTotal; ++row)
    {
        const uint32_t rowOffset = row * cols;
        T mean = T(0);
        for (uint32_t col = 0; col < cols; ++col)
        {
            mean += inPtr[rowOffset + col];
        }
        mean *= invCols;

        T var = T(0);
        for (uint32_t col = 0; col < cols; ++col)
        {
            const T diff = inPtr[rowOffset + col] - mean;
            var += diff * diff;
        }
        var *= invCols;

        const T invStd = static_cast<T>(1) / std::sqrt(var + _eps);
        const T invStd3 = invStd * invStd * invStd;

        T dVar = T(0);
        T dMean = T(0);
        T sumDiff = T(0);
        for (uint32_t col = 0; col < cols; ++col)
        {
            const T diff = inPtr[rowOffset + col] - mean;
            const T dY = dYPtr[rowOffset + col];
            const T dXhat = dY * gPtr[col];

            dVar += dXhat * diff * static_cast<T>(-0.5) * invStd3;
            dMean += dXhat * -invStd;
            sumDiff += diff;
        }
        dMean += dVar * static_cast<T>(-2) * sumDiff * invCols;

        for (uint32_t col = 0; col < cols; ++col)
        {
            const T diff = inPtr[rowOffset + col] - mean;
            const T dXhat = dYPtr[rowOffset + col] * gPtr[col];
            dXPtr[rowOffset + col] = dXhat * invStd
                                     + dVar * static_cast<T>(2) * diff * invCols
                                     + dMean * invCols;
        }
    }
}

template<class T>
void Sub(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{
    if (_out->GetShape()!=_L->GetShape() || _out->GetShape()!=_R->GetShape())
    { BroadcastBinary(_out,_L,_R,1u);return; }
    for (uint32_t i = 0; i < _R->GetElementCount(); i++)
    {
        _out->DataWrite()[i] = _L->DataRead()[i] - _R->DataRead()[i];
    }
}

template<class T>
void BroadcastAdd(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{ BroadcastBinary(_out,_L,_R,0u); }

template<class T>
void BroadcastMul(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{ BroadcastBinary(_out,_L,_R,2u); }

template<class T>
void BroadcastAddRows(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{ BroadcastBinary(_out,_L,_R,0u); }

template<class T>
void BroadcastSubRows(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{ BroadcastBinary(_out,_L,_R,1u); }

template<class T>
void BroadcastMulRows(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{ BroadcastBinary(_out,_L,_R,2u); }

template<class T>
void BroadcastAdd(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R, ThreadPool* _threadPool, uint32_t batchSize = 128)
{
    assert(_L->GetPlaneCount() == 1);
    assert(_R->GetPlaneCount() == 1);
    const uint32_t elemCount = _L->GetElementCount();
    if (elemCount == 0) return;

    const uint32_t dimsX = _L->GetWidth();

    T*       outPtr = _out->DataWrite();
    const T* lPtr   = _L->DataRead();
    const T* rPtr   = _R->DataRead();

    uint32_t i              = 0;
    const uint32_t jobCount = (elemCount + batchSize - 1) / batchSize;
    std::latch latchCounter(jobCount);

    auto worker = [outPtr, lPtr, rPtr, dimsX](uint32_t _start, uint32_t _end)
    {
        for (uint32_t k = _start; k < _end; ++k)
        {
            const uint32_t col = k % dimsX;
            outPtr[k] = lPtr[k] + rPtr[col];
        }
    };

    while (i < elemCount)
    {
        const uint32_t start = i;
        i += batchSize;
        const uint32_t end = std::min(elemCount, i);

        _threadPool->Enqueue([&latchCounter, worker, start, end]()
        {
            worker(start, end);
            latchCounter.count_down();
        });
    }

    latchCounter.wait();
}

template<class T>
void PerElementMul(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{
    for (uint32_t i = 0; i < _R->GetElementCount(); i++)
    {
        _out->DataWrite()[i] = _L->DataRead()[i] * _R->DataRead()[i];
    }
}

template<class T>
void PerElementMul(MatrixBase<T>* _out,
                   MatrixBase<T>* _L,
                   MatrixBase<T>* _R,
                   ThreadPool* _threadPool,
                   uint32_t batchSize = 128)
{
    // DataRead can map GPU staging buffers, so the input objects must be mutable.
    const uint32_t elemCount = _R->GetElementCount();
    if (elemCount == 0) return;

    T*       outPtr = _out->DataWrite();
    const T* lPtr   = _L->DataRead();
    const T* rPtr   = _R->DataRead();

    uint32_t i = 0;
    const uint32_t jobCount = (elemCount + batchSize - 1) / batchSize;
    std::latch latchCounter(jobCount);

    auto worker = [outPtr, lPtr, rPtr](uint32_t start, uint32_t end)
    {
        for (uint32_t k = start; k < end; ++k)
        {
            outPtr[k] = lPtr[k] * rPtr[k];
        }
    };

    while (i < elemCount)
    {
        const uint32_t start = i;
        i += batchSize;
        const uint32_t end = std::min(elemCount, i);

        _threadPool->Enqueue([&latchCounter, worker, start, end]()
        {
            worker(start, end);
            latchCounter.count_down();
        });
    }

    latchCounter.wait();
}


template<class T, class Func>
void MapFunction_Zero(MatrixBase<T>* _out, Func F)
{
    const uint32_t elemCount = _out->GetElementCount();
    for (uint32_t i = 0; i < elemCount; i++)
    {
        _out->DataWrite()[i] = F();
    }
}

template<class T, class U, class Func>
void MapFunction(MatrixBase<T>* _out, MatrixBase<U>* _in, Func F)
{
    const U* input = _in->DataRead();
    T* output = _out->DataWrite();
    for (uint32_t i = 0; i < _in->GetElementCount(); i++)
    {
        output[i] = F(input[i]);
    }
}

template<class T, class U, class Func>
void MapFunction(MatrixBase<T>* _out, MatrixBase<U>* _in, Func F, ThreadPool* _threadPool, uint32_t batchSize = 4096)
{
    const uint32_t elemCount = _out->GetElementCount();
    if(elemCount == 0) return;

    // GPU-backed matrices map and synchronize on the caller thread only.
    const U* inPtr = _in->DataRead();
    T* outPtr = _out->DataWrite();
    uint32_t i = 0;
    const uint32_t jobCount = elemCount % batchSize == 0 ? (elemCount / batchSize) : (elemCount / batchSize) + 1;
    std::latch latchCounter(jobCount);

    auto worker = [outPtr, inPtr, &F](uint32_t _start, uint32_t _end)
        {
            for (uint32_t k = _start; k < _end; k++)
            {
                outPtr[k] = F(inPtr[k]);
            }
        };

    while(i < elemCount)
    {
        uint32_t start = i;
        i = i + batchSize;
        uint32_t end = std::min(elemCount, i);

        _threadPool->Enqueue([&worker, start, end, &_out, &latchCounter]()
                                                {
                                                    worker(start, end);
                                                    latchCounter.count_down();
                                                });
    }

    latchCounter.wait(); 
}

template<class T, class Func>
void PerElement_Func(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R, Func F)
{
    const uint32_t elemCount = _L->GetElementCount();
    for (uint32_t i = 0; i < elemCount; i++)
    {
        _out->DataWrite()[i] = F(_L->DataWrite()[i], _R->DataWrite()[i]);
    }
}

template<class T>
void ReluMat(MatrixBase<T>* _out, MatrixBase<T>* _in)
{
    assert(_out);
    assert(_in);
    const uint32_t elemCount = _in->GetElementCount();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        const T x = _in->DataRead()[i];
        _out->DataWrite()[i] = x > static_cast<T>(0) ? x : static_cast<T>(0);
    }
}

template<class T>
void ReluDerivativeMat(MatrixBase<T>* _out, MatrixBase<T>* _in)
{
    assert(_out);
    assert(_in);
    const uint32_t elemCount = _in->GetElementCount();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        const T x = _in->DataRead()[i];
        _out->DataWrite()[i] = x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);
    }
}

template<class T>
void GeluMat(MatrixBase<T>* _out, MatrixBase<T>* _in)
{
    assert(_out);
    assert(_in);
    const uint32_t elemCount = _in->GetElementCount();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        const T x = _in->DataRead()[i];
        _out->DataWrite()[i] = static_cast<T>(0.5) * x
                               * (static_cast<T>(1) + std::erf(x / std::sqrt(static_cast<T>(2))));
    }
}

template<class T>
void GeluDerivtiveMat(MatrixBase<T>* _out, MatrixBase<T>* _in)
{
    assert(_out);
    assert(_in);
    const T sqrt2 = std::sqrt(static_cast<T>(2));
    const T sqrt2pi = std::sqrt(static_cast<T>(2) * static_cast<T>(3.14159265358979323846));
    const uint32_t elemCount = _in->GetElementCount();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        const T x = _in->DataRead()[i];
        const T erfTerm = std::erf(x / sqrt2);
        const T expTerm = std::exp(static_cast<T>(-0.5) * x * x);
        _out->DataWrite()[i] = static_cast<T>(0.5) * (static_cast<T>(1) + erfTerm)
                               + (x * expTerm) / sqrt2pi;
    }
}

// Fused activation gradient. Output storage must be distinct from both inputs.
template<class T>
void GeluBackwards(MatrixBase<T>* out, MatrixBase<T>* input, MatrixBase<T>* grad)
{
    assert(out && input && grad);
    assert(out->GetWidth() == input->GetWidth() && out->GetHeight() == input->GetHeight()
           && out->GetPlaneCount() == input->GetPlaneCount());
    assert(out->GetWidth() == grad->GetWidth() && out->GetHeight() == grad->GetHeight()
           && out->GetPlaneCount() == grad->GetPlaneCount());
    const T sqrt2 = std::sqrt(static_cast<T>(2));
    const T sqrt2pi = std::sqrt(static_cast<T>(2) * static_cast<T>(3.14159265358979323846));
    const T* xPtr = input->DataRead();
    const T* gPtr = grad->DataRead();
    T* result = out->DataWrite();
    for (uint32_t i = 0; i < input->GetElementCount(); ++i)
    {
        const T x = xPtr[i];
        const T derivative = static_cast<T>(0.5) * (static_cast<T>(1) + std::erf(x / sqrt2))
                             + x * std::exp(static_cast<T>(-0.5) * x * x) / sqrt2pi;
        result[i] = gPtr[i] * derivative;
    }
}

template<class T, class U>
void Scale(MatrixBase<T>* _out, MatrixBase<U>* _in, MatrixBase<U>* _val)
{
    assert(_val->GetWidth() == 1 && _val->GetHeight() == 1);
	T scaleval = _val->GetValue(0u, 0u);
    for (uint32_t i = 0; i < _in->GetElementCount(); i++)
    {
        _out->DataWrite()[i] = _in->DataRead()[i] * scaleval;
    }
}

template<class T, class U>
void Scale(MatrixBase<T>* _out, MatrixBase<U>* _in, float _val)
{
    for (uint32_t i = 0; i < _in->GetElementCount(); i++)
    {
        _out->DataWrite()[i] = _in->DataRead()[i] * _val;
    }
}

template<class T, class U>
void ScaleAdd(MatrixBase<T>* _out, MatrixBase<U>* _in, float _scale, MatrixBase<U>* _add)
{
    assert(_out->GetElementCount() == _in->GetElementCount());
    assert(_out->GetElementCount() == _add->GetElementCount());
    const uint32_t elemCount = _out->GetElementCount();
    const U* inPtr = _in->DataRead();
    const U* addPtr = _add->DataRead();
    T* outPtr = _out->DataWrite();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        outPtr[i] = static_cast<T>(inPtr[i] * _scale + addPtr[i]);
    }
}

template<class T, class U>
void ScaleAdd(MatrixBase<T>* _out, MatrixBase<U>* _in, MatrixBase<U>* _scale, MatrixBase<U>* _add)
{
    assert(_out->GetElementCount() == _in->GetElementCount());
    assert(_out->GetElementCount() == _add->GetElementCount());
    assert(_scale->GetElementCount() >= 1u);
    const uint32_t elemCount = _out->GetElementCount();
    const U* inPtr = _in->DataRead();
    const U* addPtr = _add->DataRead();
    const U scaleVal = _scale->DataRead()[0];
    T* outPtr = _out->DataWrite();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        outPtr[i] = static_cast<T>(inPtr[i] * scaleVal + addPtr[i]);
    }
}

template<class T>
void Clear(MatrixBase<T>* _out)
{
    memset(_out->DataWrite(), 0, sizeof(T) * _out->GetElementCount());
}

// Binary keep mask. Scaling by 1/(1-probability) remains the layer's job.
// An explicit seed reproduces the same logical mask on CPU, Metal and DX11.
template<class T>
void DropoutMask(MatrixBase<T>* out, float probability,
                 uint32_t seed = RandomUtils::NextSeed())
{
    assert(out && std::isfinite(probability));
    if (out->GetElementCount() == 0u) return;
    T* values = out->DataWrite();
    const uint32_t stream = RandomUtils::pcg_hash(seed);
    for (uint32_t i = 0; i < out->GetElementCount(); ++i)
    {
        const float uniform = float(RandomUtils::pcg_hash(i ^ stream) >> 8u) * (1.0f / 16777216.0f);
        values[i] = uniform < probability ? T(0) : T(1);
    }
}

template<class T>
void Fill(MatrixBase<T>* _out, float _val)
{
    for (uint32_t i = 0; i < _out->GetElementCount(); i++)
    {
        _out->DataWrite()[i] = _val;
    }
}

template<class T>
void Fill(MatrixBase<T>* _out, MatrixBase<T>* _val)
{
    T val = _val->GetValue(0,0);
    for (uint32_t i = 0; i < _out->GetElementCount(); i++)
    {
        _out->DataWrite()[i] = val;
    }
}

template<class T>
void ReduceSum(MatrixBase<T>* out, MatrixBase<T>* input, MatrixAxes axes)
{
    if (!out || !input || out->GetShape() != input->GetShape().Reduced(axes))
        throw std::invalid_argument("ReduceSum output must match the input with reduced axes retained.");
    const auto& in = input->GetShape();
    const auto& result = out->GetShape();
    const uint32_t bits = static_cast<uint32_t>(axes);
    const T* source = input->DataRead();
    T* destination = out->DataWrite();
    for (uint32_t b=0;b<result.batch;++b) for (uint32_t c=0;c<result.channels;++c)
    for (uint32_t y=0;y<result.height;++y) for (uint32_t x=0;x<result.width;++x)
    {
        double sum = 0.0;
        for (uint32_t rb=0;rb<((bits&1u)?in.batch:1u);++rb)
        for (uint32_t rc=0;rc<((bits&2u)?in.channels:1u);++rc)
        for (uint32_t ry=0;ry<((bits&8u)?in.height:1u);++ry)
        for (uint32_t rx=0;rx<((bits&4u)?in.width:1u);++rx)
            sum += source[in.Index((bits&1u)?rb:b,(bits&2u)?rc:c,(bits&4u)?rx:x,(bits&8u)?ry:y)];
        destination[result.Index(b,c,x,y)] = static_cast<T>(sum);
    }
}

template<class T>
void SumColumns(MatrixBase<T>* _out, MatrixBase<T>* _in)
{
    assert(_out);
    assert(_in);
    assert(_out->GetWidth() == _in->GetWidth());
    assert(_out->GetHeight() > 0);

    const uint32_t dimsX = _in->GetWidth();
    const uint32_t dimsY = _in->GetHeight();
    const T* inPtr = _in->DataRead();

    for (uint32_t col = 0; col < dimsX; ++col)
    {
        T sum = T(0);
        for (uint32_t row = 0; row < dimsY; ++row)
        {
            sum += inPtr[row * dimsX + col];
        }
        _out->SetValue(col, 0u, sum);
    }
}

template<class T>
void SumRows(MatrixBase<T>* _out, MatrixBase<T>* _in)
{
    assert(_out);
    assert(_in);
    assert(_out->GetWidth() == 1);
    assert(_out->GetHeight() == _in->GetHeight());
    assert(_out->GetPlaneCount() == 1);
    assert(_in->GetPlaneCount() == 1);

    const uint32_t rows = _in->GetHeight();
    const uint32_t cols = _in->GetWidth();
    const T* inPtr = _in->DataRead();

    for (uint32_t row = 0; row < rows; ++row)
    {
        T sum = T(0);
        const uint32_t rowOffset = row * cols;
        for (uint32_t col = 0; col < cols; ++col)
        {
            sum += inPtr[rowOffset + col];
        }
        _out->SetValue(0u, row, sum);
    }
}

template<class T>
void SumSpatialDimension(MatrixBase<T>* _out,
                         MatrixBase<T>* _in,
                         uint32_t _zLayer,
                         const MatrixCoordinate& _outCoord)
{
    assert(_out);
    assert(_in);
    assert(_zLayer < _in->GetPlaneCount());
    const auto destination=_out->GetShape().Index(_outCoord.batch,_outCoord.channel,_outCoord.x,_outCoord.y);

    const uint32_t spatialCount = _in->GetWidth() * _in->GetHeight();
    const T* inPtr = _in->DataRead();
    const uint32_t offset = _zLayer * spatialCount;
    T sum = T(0);
    for (uint32_t i = 0; i < spatialCount; ++i)
    {
        sum += inPtr[offset + i];
    }
    _out->DataWrite()[destination]=sum;
}

template< class T>
T Maximum(MatrixBase<T>* _in)
{
    T max = std::numeric_limits<T>::lowest();
    for (uint32_t i = 0; i < _in->GetElementCount(); i++)
    {
        max = _in->DataRead()[i] > max ? _in->DataRead()[i] : max;
    }
    return max;
}

template< class T>
T Minimum(MatrixBase<T>*_in)
{
    T min = std::numeric_limits<T>::max();
    for (uint32_t i = 0; i < _in->GetElementCount(); i++)
    {
        min = _in->DataRead()[i] < min ? _in->DataRead()[i] : min;
    }
    return min;
}


template<class T>
void VConcat(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R )
{
    assert(_L->GetPlaneCount() == 1);
    assert(_R->GetPlaneCount() == 1);
    assert(_L->GetWidth() == _R->GetWidth());
    assert(_L->GetHeight() + _R->GetHeight() <= _out->GetHeight());

    memcpy(_out->DataWrite(), _L->DataRead(), sizeof(T) * _L->GetElementCount());
    memcpy(_out->DataWrite() + _L->GetElementCount(), _R->DataRead(), sizeof(T) * _R->GetElementCount());
}

template<class T>
void VConcat(MatrixBase<T>* _out, std::vector<MatrixBase<T>*>& _mats )
{
    uint32_t offset = 0u;
    for(uint32_t i = 0; i < _mats.size(); i++)
    {
        MatrixBase<T>* mat = _mats[i];
        assert(mat->GetPlaneCount() == 1);
        assert(_out->GetWidth() == mat->GetWidth());

        memcpy(_out->DataWrite() + offset, mat->DataRead(), sizeof(T) * mat->GetElementCount());
        offset+=mat->GetElementCount();
    }
}

template<class T, class Mat>
void VConcat(MatrixBase<T>* _out, const std::vector<Mat*>& _mats )
{
    uint32_t offset = 0u;
    for(uint32_t i = 0; i < _mats.size(); i++)
    {
        MatrixBase<T>* mat = static_cast<MatrixBase<T>*>(_mats[i]);
        assert(mat->GetPlaneCount() == 1);
        assert(_out->GetWidth() == mat->GetWidth());

        memcpy(_out->DataWrite() + offset, mat->DataRead(), sizeof(T) * mat->GetElementCount());
        offset+=mat->GetElementCount();
    }
}

template<class T>
void HConcat(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R )
{
    assert(_L->GetPlaneCount() == 1);
    assert(_R->GetPlaneCount() == 1);
    assert(_L->GetHeight() == _R->GetHeight());
    assert(_L->GetWidth() + _R->GetWidth() <= _out->GetWidth());

    const uint32_t rows = _L->GetHeight();
    const uint32_t colsL = _L->GetWidth();
    const uint32_t colsR = _R->GetWidth();
    const uint32_t outCols = _out->GetWidth();

    T* outPtr = _out->DataWrite();
    const T* lPtr = _L->DataRead();
    const T* rPtr = _R->DataRead();

    const size_t bytesL = sizeof(T) * colsL;
    const size_t bytesR = sizeof(T) * colsR;

    for (uint32_t row = 0; row < rows; ++row)
    {
        const uint32_t lOffset = row * colsL;
        const uint32_t rOffset = row * colsR;
        const uint32_t outOffset = row * outCols;
        memcpy(outPtr + outOffset, lPtr + lOffset, bytesL);
        memcpy(outPtr + outOffset + colsL, rPtr + rOffset, bytesR);
    }
}

template<class T>
void ValidateConvolution(MatrixBase<T>* output, MatrixBase<T>* input, MatrixBase<T>* kernel,
                         uint32_t channel,uint32_t stride,uint32_t dilation)
{
    if(!output || !input || !kernel || !stride || !dilation || channel>=output->GetChannelCount()
        || output->GetBatchCount()!=input->GetBatchCount() || kernel->GetBatchCount()!=1u
        || kernel->GetChannelCount()!=input->GetChannelCount())
        throw std::invalid_argument("Invalid convolution shapes or sampling parameters.");
}

inline int64_t ConvolutionInputCoordinate(uint32_t output,uint32_t kernel,uint32_t stride,
                                         uint32_t dilation,uint32_t padding,bool transposed)
{
    const int64_t coordinate=transposed ? int64_t(output)+padding-int64_t(kernel)*dilation
                                      : int64_t(output)*stride+int64_t(kernel)*dilation-padding;
    if(transposed && (coordinate<0 || coordinate%stride)) return -1;
    return transposed ? coordinate/stride : coordinate;
}

template<class T>
void ConvolutionForward(MatrixBase<T>* output,MatrixBase<T>* input,MatrixBase<T>* kernel,
                        MatrixBase<T>* bias,uint32_t channel,uint32_t stride,uint32_t dilation,
                        uint32_t padding,bool transposed)
{
    ValidateConvolution(output,input,kernel,channel,stride,dilation);
    const auto& o=output->GetShape();const auto& i=input->GetShape();const auto& k=kernel->GetShape();
    const T* values=input->DataRead();const T* weights=kernel->DataRead();
    const T biasValue=bias ? bias->DataRead()[channel] : T(0);T* result=output->DataWrite();
    for(uint32_t b=0;b<o.batch;++b) for(uint32_t y=0;y<o.height;++y) for(uint32_t x=0;x<o.width;++x) {
        T sum=biasValue;
        for(uint32_t c=0;c<i.channels;++c) for(uint32_t ky=0;ky<k.height;++ky) for(uint32_t kx=0;kx<k.width;++kx) {
            const int64_t ix=ConvolutionInputCoordinate(x,kx,stride,dilation,padding,transposed);
            const int64_t iy=ConvolutionInputCoordinate(y,ky,stride,dilation,padding,transposed);
            if(ix>=0 && iy>=0 && ix<i.width && iy<i.height)
                sum+=values[i.Index(b,c,uint32_t(ix),uint32_t(iy))]*weights[k.Index(0,c,kx,ky)];
        }
        result[o.Index(b,channel,x,y)]=sum;
    }
}

template<class T>
void ConvolutionBackwards(MatrixBase<T>* error,MatrixBase<T>* input,MatrixBase<T>* kernel,
                          MatrixBase<T>* weightGradient,MatrixBase<T>* inputGradient,uint32_t channel,
                          uint32_t stride,uint32_t dilation,uint32_t padding,bool transposed)
{
    ValidateConvolution(error,input,kernel,channel,stride,dilation);
    if((weightGradient && weightGradient->GetShape()!=kernel->GetShape())
        || (inputGradient && inputGradient->GetShape()!=input->GetShape()))
        throw std::invalid_argument("Convolution gradients must match their parameter/input shapes.");
    const auto& o=error->GetShape();const auto& i=input->GetShape();const auto& k=kernel->GetShape();
    const T* values=input->DataRead();const T* weights=kernel->DataRead();const T* upstream=error->DataRead();
    T* dw=weightGradient ? weightGradient->DataWrite() : nullptr;
    T* dx=inputGradient ? inputGradient->DataWrite() : nullptr;
    for(uint32_t b=0;b<o.batch;++b) for(uint32_t y=0;y<o.height;++y) for(uint32_t x=0;x<o.width;++x) {
        const T grad=upstream[o.Index(b,channel,x,y)];
        for(uint32_t c=0;c<i.channels;++c) for(uint32_t ky=0;ky<k.height;++ky) for(uint32_t kx=0;kx<k.width;++kx) {
            const int64_t ix=ConvolutionInputCoordinate(x,kx,stride,dilation,padding,transposed);
            const int64_t iy=ConvolutionInputCoordinate(y,ky,stride,dilation,padding,transposed);
            if(ix>=0 && iy>=0 && ix<i.width && iy<i.height) {
                const auto pixel=i.Index(b,c,uint32_t(ix),uint32_t(iy)),weight=k.Index(0,c,kx,ky);
                if(dw) dw[weight]+=values[pixel]*grad;
                if(dx) dx[pixel]+=weights[weight]*grad;
            }
        }
    }
}

template<class T>
void Conv2DSingleChannel(MatrixBase<T>* output,MatrixBase<T>* input,MatrixBase<T>* kernel,MatrixBase<T>* bias,
                         uint32_t channel,uint32_t stride,uint32_t dilation,uint32_t padding)
{ ConvolutionForward(output,input,kernel,bias,channel,stride,dilation,padding,false); }

template<class T>
void Conv2DSingleChannelBackwards(MatrixBase<T>* error,MatrixBase<T>* input,MatrixBase<T>* kernel,
                                  MatrixBase<T>* dw,MatrixBase<T>* dx,uint32_t channel,uint32_t stride,
                                  uint32_t dilation,uint32_t padding)
{ ConvolutionBackwards(error,input,kernel,dw,dx,channel,stride,dilation,padding,false); }

template<class T>
void Conv2DTransposeSingleChannel(MatrixBase<T>* output,MatrixBase<T>* input,MatrixBase<T>* kernel,MatrixBase<T>* bias,
                                  uint32_t channel,uint32_t stride,uint32_t dilation,uint32_t padding)
{ ConvolutionForward(output,input,kernel,bias,channel,stride,dilation,padding,true); }

template<class T>
void Conv2DTransposeSingleChannelBackwards(MatrixBase<T>* error,MatrixBase<T>* input,MatrixBase<T>* kernel,
                                           MatrixBase<T>* dw,MatrixBase<T>* dx,uint32_t channel,uint32_t stride,
                                           uint32_t dilation,uint32_t padding)
{ ConvolutionBackwards(error,input,kernel,dw,dx,channel,stride,dilation,padding,true); }

template<class T>
void AdamWUpdate(MatrixBase<T>* param,
                 MatrixBase<T>* grad,
                 MatrixBase<T>* mt,
                 MatrixBase<T>* vt,
                 T lr,
                 T beta1,
                 T beta2,
                 T beta1_pow_t,
                 T beta2_pow_t,
                 T eps,
                 T weightDecay)
{
    assert(param);
    assert(grad);
    assert(mt);
    assert(vt);
    const uint32_t elemCount = param->GetElementCount();
    assert(grad->GetElementCount() == elemCount);
    assert(mt->GetElementCount() == elemCount);
    assert(vt->GetElementCount() == elemCount);

    const T* gradPtr = grad->DataRead();
    T* mtPtr = mt->DataWrite();
    T* vtPtr = vt->DataWrite();
    T* paramPtr = param->DataWrite();

    const T one = T(1);
    const T invBeta1 = one - beta1;
    const T invBeta2 = one - beta2;
    const T invBeta1Pow = one - beta1_pow_t;
    const T invBeta2Pow = one - beta2_pow_t;

    for (uint32_t i = 0; i < elemCount; ++i)
    {
        const T g = gradPtr[i];
        const T mtVal = beta1 * mtPtr[i] + invBeta1 * g;
        const T vtVal = beta2 * vtPtr[i] + invBeta2 * g * g;
        mtPtr[i] = mtVal;
        vtPtr[i] = vtVal;

        const T m_hat = mtVal / invBeta1Pow;
        const T v_hat = vtVal / invBeta2Pow;
        const T p = paramPtr[i];
        const T update = lr * m_hat / (std::sqrt(v_hat) + eps);
        const T decay = lr * weightDecay * p;
        paramPtr[i] = p + update - decay;
    }
}

template<class T>
void AdamUpdate(MatrixBase<T>* param,
                MatrixBase<T>* grad,
                MatrixBase<T>* mt,
                MatrixBase<T>* vt,
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


// VAE reparameterization backward pass.
// Computes gradients for both the mu and logvar heads given the upstream gradient dz,
// the saved mu, logvar, eps from the forward pass, and the KL weight:
//
//   muGrad[i]  = dz[i] - kl_weight * mu[i]
//   lvGrad[i]  = dz[i] * eps[i] * 0.5 * sigma  -  kl_weight * 0.5 * (sigma^2 - 1)
//
// where  lv = clamp(logvar[i], -10, 4)  and  sigma = exp(0.5 * lv).
template<class T>
void ReparameterizeBackwardsMat(MatrixBase<T>* _muGrad,  MatrixBase<T>* _lvGrad,
                                MatrixBase<T>* _dz,     MatrixBase<T>* _mu,
                                MatrixBase<T>* _logvar, MatrixBase<T>* _eps,
                                float _klWeight)
{
    assert(_muGrad && _lvGrad && _dz && _mu && _logvar && _eps);
    const size_t n       = _dz->GetElementCount();
    const T* dz          = _dz->DataRead();
    const T* muData      = _mu->DataRead();
    const T* lvData      = _logvar->DataRead();
    const T* epsData     = _eps->DataRead();
    T* muGrad            = _muGrad->DataWrite();
    T* lvGrad            = _lvGrad->DataWrite();
    for (size_t i = 0; i < n; ++i)
    {
        const float lv    = std::max(-10.0f, std::min(4.0f, static_cast<float>(lvData[i])));
        const float sigma = std::exp(0.5f * lv);
        muGrad[i] = dz[i] - static_cast<T>(_klWeight * static_cast<float>(muData[i]));
        lvGrad[i] = static_cast<T>(
            static_cast<float>(dz[i]) * static_cast<float>(epsData[i]) * 0.5f * sigma
            - _klWeight * 0.5f * (sigma * sigma - 1.0f));
    }
}

// Reparameterization trick: z = mu + exp(0.5 * clamp(logvar, -10, 4)) * eps,  eps ~ N(0,I).
// Writes both _latent (z) and _eps so the backward pass can reuse eps without re-sampling.
template<class T>
void ReparameterizeMat(MatrixBase<T>* _latent, MatrixBase<T>* _eps,
                       MatrixBase<T>* _mu,     MatrixBase<T>* _logvar)
{
    assert(_latent && _eps && _mu && _logvar);
    thread_local std::mt19937 s_rng{std::random_device{}()};
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const size_t n       = _mu->GetElementCount();
    const T* muData      = _mu->DataRead();
    const T* lvData      = _logvar->DataRead();
    T* latentData        = _latent->DataWrite();
    T* epsData           = _eps->DataWrite();
    for (size_t i = 0; i < n; ++i)
    {
        const float e     = dist(s_rng);
        const float lv    = std::max(-10.0f, std::min(4.0f, static_cast<float>(lvData[i])));
        const float sigma = std::exp(0.5f * lv);
        epsData[i]    = static_cast<T>(e);
        latentData[i] = muData[i] + static_cast<T>(sigma * e);
    }
}

#include "MatrixBase_Mul.h"

// The caller supplies each unique parameter gradient once, after averaging.
// GPU overloads keep reduction and scaling on the device.
template<class T, class Mat>
void ClipGradientNorm(const std::vector<Mat*>& gradients, float maximumNorm,
                      GradientClipWorkspace<T,Mat>& workspace)
{
    (void)workspace;
    if (!std::isfinite(maximumNorm) || maximumNorm < 0.0f)
        throw std::invalid_argument("Gradient clip norm must be finite and nonnegative.");
    if (maximumNorm == 0.0f) return;
    double squaredNorm = 0.0;
    for (auto* gradient : gradients)
    {
        if (!gradient) throw std::invalid_argument("Null gradient in norm clipping.");
        if (gradient->GetElementCount() == 0u) continue;
        const T* values = gradient->DataRead();
        for (uint32_t i = 0; i < gradient->GetElementCount(); ++i)
        {
            const double value = static_cast<double>(values[i]);
            squaredNorm += value * value;
        }
        if constexpr (requires { gradient->Unmap(); }) gradient->Unmap();
    }
    if (!std::isfinite(squaredNorm))
        throw std::runtime_error("Cannot clip non-finite gradients.");
    const float factor = float(std::min(1.0,double(maximumNorm)/(std::sqrt(squaredNorm)+1e-6)));
    if (factor < 1.0f)
        for (auto* gradient : gradients)
            if (gradient->GetElementCount() != 0u) Scale(gradient,gradient,factor);
}
