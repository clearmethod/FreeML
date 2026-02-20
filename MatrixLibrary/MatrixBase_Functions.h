#pragma once

#include <ToolsLibrary/ThreadPool.h>

#include <latch>
#include <cassert>
#include <cstring>
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
void Softmax(MatrixBase<T>* out, MatrixBase<T>* in)
{
    assert(out->GetDimsX() == in->GetDimsX() && out->GetDimsY() == in->GetDimsY());
    const uint32_t rows = in->GetDimsY();
    const uint32_t cols = in->GetDimsX();
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
    assert(out->GetDimsX() == softmaxOut->GetDimsX() && out->GetDimsY() == softmaxOut->GetDimsY());
    assert(gradIn->GetDimsX() == softmaxOut->GetDimsX() && gradIn->GetDimsY() == softmaxOut->GetDimsY());

    const uint32_t rows = softmaxOut->GetDimsY();
    const uint32_t cols = softmaxOut->GetDimsX();
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

enum class TriangleDirection { Lower, Upper };
template<class T>
void TriangleMatrix(MatrixBase<T>* _out, TriangleDirection _dir, T trueVal, T falseVal)
{
    assert(_out);

    const uint32_t rows = _out->GetDimsY();
    const uint32_t cols = _out->GetDimsX();
    const T* inPtr = _out->DataRead();
    T* outPtr = _out->DataWrite();

    if (_dir == TriangleDirection::Lower)
    {
        for (uint32_t row = 0; row < rows; ++row)
        {
            const uint32_t rowOffset = row * cols;
            const uint32_t keep = row + 1 < cols ? row + 1 : cols;
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
        const uint32_t start = row < cols ? row : cols;
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

template<class T>
void GatherRows(MatrixBase<T>* _out, MatrixBase<T>* _src, MatrixBase<T>* _indices)
{
    assert(_out);
    assert(_src);
    assert(_indices);
    assert(_indices->GetDimsX() == 1u);

    const uint32_t rowWidth = _src->GetDimsX();
    const uint32_t outRows = _out->GetDimsY();
    const uint32_t outBatch = _out->GetDimsZ();
    const uint32_t srcRows = _src->GetDimsY();
    const uint32_t srcBatch = _src->GetDimsZ();

    assert(_out->GetDimsX() == rowWidth);
    assert(_indices->GetDimsY() == outRows);
    assert(_indices->GetDimsZ() == outBatch);
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
    assert(_indices->GetDimsX() == 1u);
    assert(_src->GetDimsX() == _out->GetDimsX());
    assert(_src->GetDimsY() == _indices->GetDimsY());
    assert(_src->GetDimsZ() == _indices->GetDimsZ());
    assert(_out->GetDimsZ() == 1u || _out->GetDimsZ() == _src->GetDimsZ());

    const uint32_t rowWidth = _src->GetDimsX();
    const uint32_t srcRows = _src->GetDimsY();
    const uint32_t srcBatch = _src->GetDimsZ();
    const uint32_t outRows = _out->GetDimsY();
    const uint32_t outBatch = _out->GetDimsZ();

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
    assert(_indices->GetDimsX() == 1u);
    assert(_src->GetDimsX() == _out->GetDimsX());
    assert(_src->GetDimsY() == _indices->GetDimsY());
    assert(_src->GetDimsZ() == _indices->GetDimsZ());
    assert(_out->GetDimsZ() == 1u || _out->GetDimsZ() == _src->GetDimsZ());

    const uint32_t rowWidth = _src->GetDimsX();
    const uint32_t srcRows = _src->GetDimsY();
    const uint32_t srcBatch = _src->GetDimsZ();
    const uint32_t outRows = _out->GetDimsY();
    const uint32_t outBatch = _out->GetDimsZ();

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

    const uint32_t dC = _q->GetDimsX();
    const uint32_t rows = _q->GetDimsY() * _q->GetDimsZ();
    const uint32_t qCount = _q->GetElementCount();
    if (qCount == 0u)
    {
        return;
    }

    assert(_k->GetElementCount() == qCount);
    assert(_v->GetElementCount() == qCount);
    assert(_packed->GetElementCount() == qCount * 3u);
    assert(_packed->GetDimsX() == dC * 3u);
    assert((_packed->GetDimsY() * _packed->GetDimsZ()) == rows);

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

    const uint32_t qkvCols = 3u * dC;
    const uint32_t headPlane = headDim * tokens;

    const T* qPtr = _qHeads->DataRead();
    const T* kPtr = _kHeads->DataRead();
    const T* vPtr = _vHeads->DataRead();
    T* packedPtr = _packedQKV->DataWrite();

    for (uint32_t row = 0; row < tokens; ++row)
    {
        const uint32_t rowPackedBase = row * qkvCols;
        const uint32_t rowHeadOffset = row * headDim;
        for (uint32_t head = 0; head < nHead; ++head)
        {
            const uint32_t srcOffset = head * headPlane + rowHeadOffset;
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
    assert(_out->GetDimsX() == _in->GetDimsY());
    assert(_out->GetDimsY() == _in->GetDimsX());
    assert(_out->GetDimsZ() == _in->GetDimsZ());

    const uint32_t inX = _in->GetDimsX();
    const uint32_t inY = _in->GetDimsY();
    const uint32_t inZ = _in->GetDimsZ();
    const uint32_t outX = _out->GetDimsX();
    const uint32_t outY = _out->GetDimsY();

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
void Add(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{
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
                 T _eps = static_cast<T>(1e-5))
{
    assert(_out);
    assert(_in);
    assert(_gamma);
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

    const uint32_t cols = _in->GetDimsX();
    const uint32_t rows = _in->GetDimsY() * _in->GetDimsZ();
    const T invCols = static_cast<T>(1) / static_cast<T>(cols);

    const T* inPtr = _in->DataRead();
    const T* gPtr = _gamma->DataRead();
    const T* bPtr = _beta ? _beta->DataRead() : nullptr;
    T* outPtr = _out->DataWrite();
    T* xHatPtr = _xHat ? _xHat->DataWrite() : nullptr;

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
    assert(_gamma->GetElementCount() >= _in->GetDimsX());
    assert(_dGamma->GetDimsX() == _in->GetDimsX());
    assert(_dGamma->GetElementCount() >= _in->GetDimsX());
    if (_dBeta)
    {
        assert(_dBeta->GetDimsX() == _in->GetDimsX());
        assert(_dBeta->GetElementCount() >= _in->GetDimsX());
    }
    if (_xHat)
    {
        assert(_xHat->GetDimsX() == _in->GetDimsX());
        assert(_xHat->GetDimsY() == _in->GetDimsY());
        assert(_xHat->GetDimsZ() == _in->GetDimsZ());
    }

    const uint32_t rowsTotal = _in->GetDimsY() * _in->GetDimsZ();
    const uint32_t cols = _in->GetDimsX();
    const T invCols = static_cast<T>(1) / static_cast<T>(cols);

    const T* inPtr = _in->DataRead();
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
    for (uint32_t i = 0; i < _R->GetElementCount(); i++)
    {
        _out->DataWrite()[i] = _L->DataRead()[i] - _R->DataRead()[i];
    }
}

template<class T>
void BroadcastAdd(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);
    const uint32_t elemCount = _L->GetElementCount();
    const uint32_t dimsX = _L->GetDimsX();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        const uint32_t col = i % dimsX;
        const T lhs = _L->DataRead()[i];
        const T rhs = _R->DataRead()[col];
        _out->DataWrite()[i] = lhs + rhs;
    }
}

template<class T>
void BroadcastMul(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);
    assert(_R->GetDimsX() == _L->GetDimsX());
    assert(_R->GetDimsY() == 1);
    const uint32_t elemCount = _L->GetElementCount();
    const uint32_t dimsX = _L->GetDimsX();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        const uint32_t col = i % dimsX;
        const T lhs = _L->DataRead()[i];
        const T rhs = _R->DataRead()[col];
        _out->DataWrite()[i] = lhs * rhs;
    }
}

template<class T>
void BroadcastAddRows(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);
    assert(_R->GetDimsX() == 1);
    assert(_R->GetDimsY() == _L->GetDimsY());
    const uint32_t elemCount = _L->GetElementCount();
    const uint32_t dimsX = _L->GetDimsX();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        const uint32_t row = i / dimsX;
        const T lhs = _L->DataRead()[i];
        const T rhs = _R->DataRead()[row];
        _out->DataWrite()[i] = lhs + rhs;
    }
}

template<class T>
void BroadcastSubRows(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);
    assert(_R->GetDimsX() == 1);
    assert(_R->GetDimsY() == _L->GetDimsY());
    const uint32_t elemCount = _L->GetElementCount();
    const uint32_t dimsX = _L->GetDimsX();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        const uint32_t row = i / dimsX;
        const T lhs = _L->DataRead()[i];
        const T rhs = _R->DataRead()[row];
        _out->DataWrite()[i] = lhs - rhs;
    }
}

template<class T>
void BroadcastMulRows(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R)
{
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);
    assert(_R->GetDimsX() == 1);
    assert(_R->GetDimsY() == _L->GetDimsY());
    const uint32_t elemCount = _L->GetElementCount();
    const uint32_t dimsX = _L->GetDimsX();
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        const uint32_t row = i / dimsX;
        const T lhs = _L->DataRead()[i];
        const T rhs = _R->DataRead()[row];
        _out->DataWrite()[i] = lhs * rhs;
    }
}

template<class T>
void BroadcastAdd(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R, ThreadPool* _threadPool, uint32_t batchSize = 128)
{
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);
    const uint32_t elemCount = _L->GetElementCount();
    if (elemCount == 0) return;

    const uint32_t dimsX = _L->GetDimsX();

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
                   const MatrixBase<T>* _L,
                   const MatrixBase<T>* _R,
                   ThreadPool* _threadPool,
                   uint32_t batchSize = 128)
{
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
    for (uint32_t i = 0; i < _in->GetElementCount(); i++)
    {
        _out->DataWrite()[i] = F(_in->DataWrite()[i]);
    }
}

template<class T, class U, class Func>
void MapFunction(MatrixBase<T>* _out, MatrixBase<U>* _in, Func F, ThreadPool* _threadPool, uint32_t batchSize = 4096)
{
    const uint32_t elemCount = _out->GetElementCount();
    if(elemCount == 0) return;

    uint32_t i = 0;
    const uint32_t jobCount = elemCount % batchSize == 0 ? (elemCount / batchSize) : (elemCount / batchSize) + 1;
    std::latch latchCounter(jobCount);

    auto worker = [&latchCounter, &_out, &_in, &F](uint32_t _start, uint32_t _end)
        {
            for (uint32_t k = _start; k < _end; k++)
            {
                _out->DataWrite()[k] = F(_in->DataRead()[k]);
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

template<class T, class U>
void Scale(MatrixBase<T>* _out, MatrixBase<U>* _in, MatrixBase<U>* _val)
{
    assert(_val->GetDimsX() == 1 && _val->GetDimsY() == 1);
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
void SumColumns(MatrixBase<T>* _out, MatrixBase<T>* _in)
{
    assert(_out);
    assert(_in);
    assert(_out->GetDimsX() == _in->GetDimsX());
    assert(_out->GetDimsY() > 0);

    const uint32_t dimsX = _in->GetDimsX();
    const uint32_t dimsY = _in->GetDimsY();
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
    assert(_out->GetDimsX() == 1);
    assert(_out->GetDimsY() == _in->GetDimsY());
    assert(_out->GetDimsZ() == 1);
    assert(_in->GetDimsZ() == 1);

    const uint32_t rows = _in->GetDimsY();
    const uint32_t cols = _in->GetDimsX();
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
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);
    assert(_L->GetDimsX() == _R->GetDimsX());
    assert(_L->GetDimsY() + _R->GetDimsY() <= _out->GetDimsY());

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
        assert(mat->GetDimsZ() == 1);
        assert(_out->GetDimsX() == mat->GetDimsX());

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
        assert(mat->GetDimsZ() == 1);
        assert(_out->GetDimsX() == mat->GetDimsX());

        memcpy(_out->DataWrite() + offset, mat->DataRead(), sizeof(T) * mat->GetElementCount());
        offset+=mat->GetElementCount();
    }
}

template<class T>
void HConcat(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R )
{
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);
    assert(_L->GetDimsY() == _R->GetDimsY());
    assert(_L->GetDimsX() + _R->GetDimsX() <= _out->GetDimsX());

    const uint32_t rows = _L->GetDimsY();
    const uint32_t colsL = _L->GetDimsX();
    const uint32_t colsR = _R->GetDimsX();
    const uint32_t outCols = _out->GetDimsX();

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


#include "MatrixBase_Mul.h"



