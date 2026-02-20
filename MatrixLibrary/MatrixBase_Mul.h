#pragma once

#include <cstdint>
#include <memory>

#if defined(_MSC_VER)
#define T_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define T_RESTRICT __restrict__
#else
#define T_RESTRICT
#endif

template<typename T>
inline T* AssumeAligned64IfPossible(T* ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 63u) == 0u ? std::assume_aligned<64>(ptr) : ptr;
}

template<typename T>
inline const T* AssumeAligned64IfPossible(const T* ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 63u) == 0u ? std::assume_aligned<64>(ptr) : ptr;
}

// Transpose mode
enum class TransposeMode { None, Left, Right, Both };

// Naive direct single thread mul
// Lots of dynamic allocation getting rows and columns - very slow.
// Good for debugging issues if needed...

// Naive direct single thread mul
// No dynamic alloc.
// Specify the transpose read in the template if needed. 
template<TransposeMode Mode = TransposeMode::None, class T>
void MulNoDynamicAlloc(MatrixBase<T>* out,
                       MatrixBase<T>* L,
                       MatrixBase<T>* R)
{
    assert(L->GetDimsZ() == 1);
    assert(R->GetDimsZ() == 1);

    constexpr bool TL = (Mode == TransposeMode::Left) || (Mode == TransposeMode::Both);
    constexpr bool TR = (Mode == TransposeMode::Right) || (Mode == TransposeMode::Both);

    const uint32_t L_rows = L->GetDimsY();
    const uint32_t L_cols = L->GetDimsX();
    const uint32_t R_rows = R->GetDimsY();
    const uint32_t R_cols = R->GetDimsX();

    const uint32_t M = TL ? L_cols : L_rows;
    const uint32_t K = TL ? L_rows : L_cols;
    const uint32_t N = TR ? R_rows : R_cols;

    assert(K == (TR ? R_cols : R_rows));
    assert(out->GetDimsY() == M);
    assert(out->GetDimsX() == N);

    const T* T_RESTRICT A = AssumeAligned64IfPossible(L->DataRead());
    const T* T_RESTRICT B = AssumeAligned64IfPossible(R->DataRead());
    T*       T_RESTRICT C = AssumeAligned64IfPossible(out->DataWrite());

    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < N; ++j)
        {
            T sum = T(0);
            for (uint32_t k = 0; k < K; ++k)
            {
                const T a = [] (const T* A, uint32_t i, uint32_t k, uint32_t L_cols) {
                    if constexpr (TL) return A[k * L_cols + i];
                    else               return A[i * L_cols + k];
                }(A, i, k, L_cols);

                const T b = [] (const T* B, uint32_t k, uint32_t j, uint32_t R_cols) {
                    if constexpr (TR) return B[j * R_cols + k];
                    else               return B[k * R_cols + j];
                }(B, k, j, R_cols);

                sum += a * b;
            }
            C[i * N + j] = sum;
        }
    }
}

template<TransposeMode Mode = TransposeMode::None, class T>
void MulStrided(const T* A,
                uint32_t A_rows,
                uint32_t A_cols,
                uint32_t A_rowStride,
                uint32_t A_colStride,
                const T* B,
                uint32_t B_rows,
                uint32_t B_cols,
                uint32_t B_rowStride,
                uint32_t B_colStride,
                T* C,
                uint32_t C_rowStride,
                uint32_t C_colStride)
{
    assert(A);
    assert(B);
    assert(C);

    constexpr bool TL = (Mode == TransposeMode::Left) || (Mode == TransposeMode::Both);
    constexpr bool TR = (Mode == TransposeMode::Right) || (Mode == TransposeMode::Both);

    const uint32_t M = TL ? A_cols : A_rows;
    const uint32_t K = TL ? A_rows : A_cols;
    const uint32_t N = TR ? B_rows : B_cols;

    assert(K == (TR ? B_cols : B_rows));

    auto getA = [A, A_rowStride, A_colStride](uint32_t i, uint32_t k)
    {
        if constexpr (TL)
            return A[k * A_rowStride + i * A_colStride];
        else
            return A[i * A_rowStride + k * A_colStride];
    };

    auto getB = [B, B_rowStride, B_colStride](uint32_t k, uint32_t j)
    {
        if constexpr (TR)
            return B[j * B_rowStride + k * B_colStride];
        else
            return B[k * B_rowStride + j * B_colStride];
    };

    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < N; ++j)
        {
            T sum = T(0);
            for (uint32_t k = 0; k < K; ++k)
            {
                sum += getA(i, k) * getB(k, j);
            }
            C[i * C_rowStride + j * C_colStride] = sum;
        }
    }
}

template<TransposeMode Mode = TransposeMode::None, class T>
void MulStrided(MatrixBase<T>* out,
                MatrixBase<T>* L,
                MatrixBase<T>* R,
                uint32_t L_rowStride,
                uint32_t L_colStride,
                uint32_t R_rowStride,
                uint32_t R_colStride,
                uint32_t outRowStride,
                uint32_t outColStride)
{
    assert(L->GetDimsZ() == 1);
    assert(R->GetDimsZ() == 1);

    constexpr bool TL = (Mode == TransposeMode::Left) || (Mode == TransposeMode::Both);
    constexpr bool TR = (Mode == TransposeMode::Right) || (Mode == TransposeMode::Both);

    const uint32_t L_rows = L->GetDimsY();
    const uint32_t L_cols = L->GetDimsX();
    const uint32_t R_rows = R->GetDimsY();
    const uint32_t R_cols = R->GetDimsX();

    const uint32_t M = TL ? L_cols : L_rows;
    const uint32_t K = TL ? L_rows : L_cols;
    const uint32_t N = TR ? R_rows : R_cols;

    assert(K == (TR ? R_cols : R_rows));
    assert(out->GetDimsY() == M);
    assert(out->GetDimsX() == N);

    const T* A = L->DataRead();
    const T* B = R->DataRead();
    T* C = out->DataWrite();

    MulStrided<Mode>(A,
                     L_rows,
                     L_cols,
                     L_rowStride,
                     L_colStride,
                     B,
                     R_rows,
                     R_cols,
                     R_rowStride,
                     R_colStride,
                     C,
                     outRowStride,
                     outColStride);
}

template<TransposeMode Mode = TransposeMode::None, class T>
void MulStridedBlock(const T* A,
                     uint32_t A_rows,
                     uint32_t A_cols,
                     uint32_t A_rowStride,
                     uint32_t A_colStride,
                     const T* B,
                     uint32_t B_rows,
                     uint32_t B_cols,
                     uint32_t B_rowStride,
                     uint32_t B_colStride,
                     T* C,
                     uint32_t C_rowStride,
                     uint32_t C_colStride,
                     uint32_t blockSize)
{
    assert(A);
    assert(B);
    assert(C);

    constexpr bool TL = (Mode == TransposeMode::Left) || (Mode == TransposeMode::Both);
    constexpr bool TR = (Mode == TransposeMode::Right) || (Mode == TransposeMode::Both);

    const uint32_t M = TL ? A_cols : A_rows;
    const uint32_t K = TL ? A_rows : A_cols;
    const uint32_t N = TR ? B_rows : B_cols;

    assert(K == (TR ? B_cols : B_rows));

    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < N; ++j)
        {
            C[i * C_rowStride + j * C_colStride] = T(0);
        }
    }

    auto getA = [A, A_rowStride, A_colStride](uint32_t i, uint32_t k)
    {
        if constexpr (TL)
            return A[k * A_rowStride + i * A_colStride];
        else
            return A[i * A_rowStride + k * A_colStride];
    };

    auto getB = [B, B_rowStride, B_colStride](uint32_t k, uint32_t j)
    {
        if constexpr (TR)
            return B[j * B_rowStride + k * B_colStride];
        else
            return B[k * B_rowStride + j * B_colStride];
    };

    for (uint32_t i0 = 0; i0 < M; i0 += blockSize)
    {
        for (uint32_t j0 = 0; j0 < N; j0 += blockSize)
        {
            for (uint32_t k0 = 0; k0 < K; k0 += blockSize)
            {
                const uint32_t iLimit = std::min(i0 + blockSize, M);
                const uint32_t jLimit = std::min(j0 + blockSize, N);
                const uint32_t kLimit = std::min(k0 + blockSize, K);
                for (uint32_t i = i0; i < iLimit; ++i)
                {
                    for (uint32_t j = j0; j < jLimit; ++j)
                    {
                        T sum = C[i * C_rowStride + j * C_colStride];
                        for (uint32_t k = k0; k < kLimit; ++k)
                        {
                            sum += getA(i, k) * getB(k, j);
                        }
                        C[i * C_rowStride + j * C_colStride] = sum;
                    }
                }
            }
        }
    }
}

template<TransposeMode Mode = TransposeMode::None, class T>
void MulStridedBlock(MatrixBase<T>* out,
                     MatrixBase<T>* L,
                     MatrixBase<T>* R,
                     uint32_t L_rowStride,
                     uint32_t L_colStride,
                     uint32_t R_rowStride,
                     uint32_t R_colStride,
                     uint32_t outRowStride,
                     uint32_t outColStride,
                     uint32_t blockSize,
                     uint32_t numThreads,
                     ThreadPool* pool = nullptr)
{
    assert(L->GetDimsZ() == 1);
    assert(R->GetDimsZ() == 1);

    constexpr bool TL = (Mode == TransposeMode::Left) || (Mode == TransposeMode::Both);
    constexpr bool TR = (Mode == TransposeMode::Right) || (Mode == TransposeMode::Both);

    const uint32_t L_rows = L->GetDimsY();
    const uint32_t L_cols = L->GetDimsX();
    const uint32_t R_rows = R->GetDimsY();
    const uint32_t R_cols = R->GetDimsX();

    const uint32_t M = TL ? L_cols : L_rows;
    const uint32_t K = TL ? L_rows : L_cols;
    const uint32_t N = TR ? R_rows : R_cols;

    assert(K == (TR ? R_cols : R_rows));
    assert(out->GetDimsY() == M);
    assert(out->GetDimsX() == N);

    const T* A = L->DataRead();
    const T* B = R->DataRead();
    T* C = out->DataWrite();

    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < N; ++j)
        {
            C[i * outRowStride + j * outColStride] = T(0);
        }
    }

    auto getA = [A, L_rowStride, L_colStride](uint32_t i, uint32_t k)
    {
        if constexpr (TL)
            return A[k * L_rowStride + i * L_colStride];
        else
            return A[i * L_rowStride + k * L_colStride];
    };

    auto getB = [B, R_rowStride, R_colStride](uint32_t k, uint32_t j)
    {
        if constexpr (TR)
            return B[j * R_rowStride + k * R_colStride];
        else
            return B[k * R_rowStride + j * R_colStride];
    };

    auto worker = [&](uint32_t startRow, uint32_t endRow)
    {
        for (uint32_t i0 = startRow; i0 < endRow; i0 += blockSize)
        {
            for (uint32_t j0 = 0; j0 < N; j0 += blockSize)
            {
                for (uint32_t k0 = 0; k0 < K; k0 += blockSize)
                {
                    const uint32_t iLimit = std::min(i0 + blockSize, endRow);
                    const uint32_t jLimit = std::min(j0 + blockSize, N);
                    const uint32_t kLimit = std::min(k0 + blockSize, K);
                    for (uint32_t i = i0; i < iLimit; ++i)
                    {
                        for (uint32_t j = j0; j < jLimit; ++j)
                        {
                            T sum = C[i * outRowStride + j * outColStride];
                            for (uint32_t k = k0; k < kLimit; ++k)
                            {
                                sum += getA(i, k) * getB(k, j);
                            }
                            C[i * outRowStride + j * outColStride] = sum;
                        }
                    }
                }
            }
        }
    };

    uint32_t rowsPerThread = (M + numThreads - 1) / numThreads;
    std::vector<std::pair<uint32_t,uint32_t>> jobs;
    for (uint32_t t = 0; t < numThreads; ++t)
    {
        uint32_t start = t * rowsPerThread;
        if (start >= M) break;
        uint32_t end = std::min(start + rowsPerThread, M);
        jobs.emplace_back(start, end);
    }

    if (pool == nullptr)
    {
        std::vector<std::thread> threads;
        threads.reserve(jobs.size());
        for (const auto& job : jobs)
        {
            threads.emplace_back(worker, job.first, job.second);
        }
        for (auto& t : threads)
        {
            if (t.joinable())
            {
                t.join();
            }
        }
    }
    else
    {
        std::latch latchCounter(jobs.size());
        for (const auto& job : jobs)
        {
            uint32_t start = job.first;
            uint32_t end = job.second;
            pool->Enqueue([&, start, end]
            {
                worker(start, end);
                latchCounter.count_down();
            });
        }
        latchCounter.wait();
    }
}

// Naive block based Mul
// Specify the transpose read in the template if needed.
template<TransposeMode Mode = TransposeMode::None, class T>
void MulBlock(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R, uint32_t blockSize)
{
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);

    Clear(_out);

    constexpr bool TL = (Mode == TransposeMode::Left) || (Mode == TransposeMode::Both);
    constexpr bool TR = (Mode == TransposeMode::Right) || (Mode == TransposeMode::Both);

    const uint32_t L_rows = _L->GetDimsY();
    const uint32_t L_cols = _L->GetDimsX();
    const uint32_t R_rows = _R->GetDimsY();
    const uint32_t R_cols = _R->GetDimsX();

    const uint32_t M = TL ? L_cols : L_rows;
    const uint32_t K = TL ? L_rows : L_cols;
    const uint32_t N = TR ? R_rows : R_cols;

    assert(K == (TR ? R_cols : R_rows));
    assert(_out->GetDimsY() == M);
    assert(_out->GetDimsX() == N);

    const T* T_RESTRICT A = AssumeAligned64IfPossible(_L->DataRead());
    const T* T_RESTRICT B = AssumeAligned64IfPossible(_R->DataRead());
    T* T_RESTRICT C        = AssumeAligned64IfPossible(_out->DataWrite());

    auto getA = [A, L_cols](uint32_t i, uint32_t k)
    {
        if constexpr (TL)
            return A[k * L_cols + i];
        else
            return A[i * L_cols + k];
    };

    auto getB = [B, R_cols](uint32_t k, uint32_t j)
    {
        if constexpr (TR)
            return B[j * R_cols + k];
        else
            return B[k * R_cols + j];
    };

    for (uint32_t i0 = 0; i0 < M; i0 += blockSize)
    {
        for (uint32_t j0 = 0; j0 < N; j0 += blockSize)
        {
            for (uint32_t k0 = 0; k0 < K; k0 += blockSize)
            {
                for (uint32_t i = i0; i < std::min(i0 + blockSize, M); ++i)
                {
                    for (uint32_t j = j0; j < std::min(j0 + blockSize, N); ++j)
                    {
                        T sum = C[i * N + j];
                        for (uint32_t k = k0; k < std::min(k0 + blockSize, K); ++k)
                        {
                            sum += getA(i, k) * getB(k, j);
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

// Naive block based Mul (threaded)
// Specify the transpose read in the template if needed.
template<TransposeMode Mode = TransposeMode::None, class T>
void MulBlock(MatrixBase<T>* _out,  MatrixBase<T>* _L,  MatrixBase<T>* _R,
                      uint32_t blockSize, uint32_t numThreads, ThreadPool* pool = nullptr)
{
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);
    Clear(_out);

    constexpr bool TL = (Mode == TransposeMode::Left) || (Mode == TransposeMode::Both);
    constexpr bool TR = (Mode == TransposeMode::Right) || (Mode == TransposeMode::Both);

    const uint32_t L_rows = _L->GetDimsY();
    const uint32_t L_cols = _L->GetDimsX();
    const uint32_t R_rows = _R->GetDimsY();
    const uint32_t R_cols = _R->GetDimsX();

    const uint32_t M = TL ? L_cols : L_rows;
    const uint32_t K = TL ? L_rows : L_cols;
    const uint32_t N = TR ? R_rows : R_cols;

    assert(K == (TR ? R_cols : R_rows));
    assert(_out->GetDimsY() == M);
    assert(_out->GetDimsX() == N);

    const T* T_RESTRICT A = AssumeAligned64IfPossible(_L->DataRead());
    const T* T_RESTRICT B = AssumeAligned64IfPossible(_R->DataRead());
    T*       T_RESTRICT C = AssumeAligned64IfPossible(_out->DataWrite());

    auto getA = [A, L_cols](uint32_t i, uint32_t k)
    {
        if constexpr (TL)
            return A[k * L_cols + i];
        else
            return A[i * L_cols + k];
    };

    auto getB = [B, R_cols](uint32_t k, uint32_t j)
    {
        if constexpr (TR)
            return B[j * R_cols + k];
        else
            return B[k * R_cols + j];
    };

    auto worker = [&](uint32_t startRow, uint32_t endRow)
    {
        for (uint32_t i0 = startRow; i0 < endRow; i0 += blockSize)
        {
            for (uint32_t j0 = 0; j0 < N; j0 += blockSize)
            {
                for (uint32_t k0 = 0; k0 < K; k0 += blockSize)
                {
                    const uint32_t iLimit = std::min(i0 + blockSize, endRow);
                    for (uint32_t i = i0; i < iLimit; ++i)
                    {
                        for (uint32_t j = j0; j < std::min(j0 + blockSize, N); ++j)
                        {
                            T sum = C[i * N + j];
                            for (uint32_t k = k0; k < std::min(k0 + blockSize, K); ++k)
                            {
                                sum += getA(i, k) * getB(k, j);
                            }
                            C[i * N + j] = sum;
                        }
                    }
                }
            }
        }
    };

    uint32_t rowsPerThread = (M + numThreads - 1) / numThreads;

    std::vector<std::pair<uint32_t,uint32_t>> jobs;
    for (uint32_t t = 0; t < numThreads; ++t)
    {
        uint32_t start = t * rowsPerThread;
        if(start >= M) break;
        uint32_t end = std::min(start + rowsPerThread, M);
        jobs.emplace_back(start, end);
    }

    if(pool == nullptr)
    {
        std::vector<std::thread> threads;
        threads.reserve(jobs.size());
        for(const auto& job : jobs)
        {
            threads.emplace_back(worker, job.first, job.second);
        }
        for(auto& t : threads)
        {
            if(t.joinable())
            {
                t.join();
            }
        }
    }
    else
    {
        std::latch latchCounter(jobs.size());
        for(const auto& job : jobs)
        {
            uint32_t start = job.first;
            uint32_t end = job.second;
            pool->Enqueue([&, start, end]
            {
                worker(start, end);
                latchCounter.count_down();
            });
        }
        latchCounter.wait();
    }
}

template<TransposeMode Mode = TransposeMode::None, class T>
void Mul(MatrixBase<T>* _out, MatrixBase<T>* _L, MatrixBase<T>* _R,
                      uint32_t blockSize = 32, uint32_t numThreads = 12, ThreadPool* pool = nullptr)
{
    if(!pool)
    {
        pool = GetGlobalThreadPool();
		numThreads = pool ? pool->GetThreadCount() : 1;
    }
    
    assert(_L->GetDimsZ() == 1);
    assert(_R->GetDimsZ() == 1);
    if(  _L->GetDimsX() > 32 || _R->GetDimsX() > 32
      || _L->GetDimsY() > 32 || _R->GetDimsY() > 32)
    {
        if(pool)
            MulBlock<Mode, T>(_out, _L, _R, blockSize, numThreads, pool);
        else
            MulBlock<Mode, T>(_out, _L, _R, blockSize);
    }
    else
    {
        MulNoDynamicAlloc<Mode, T>(_out, _L, _R);
    }
}

template<TransposeMode Mode = TransposeMode::None, class T>
void MatMul_Strided(MatrixBase<T>* _out,
                    MatrixBase<T>* _L,
                    MatrixBase<T>* _R,
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
    if (!pool)
    {
        pool = GetGlobalThreadPool();
        numThreads = pool ? pool->GetThreadCount() : 1;
    }

    assert(_L->GetDimsZ() == _R->GetDimsZ());
    assert(_out->GetDimsZ() == _L->GetDimsZ());

    const uint32_t L_rows = _L->GetDimsY();
    const uint32_t L_cols = _L->GetDimsX();
    const uint32_t R_rows = _R->GetDimsY();
    const uint32_t R_cols = _R->GetDimsX();
    const uint32_t outRows = _out->GetDimsY();
    const uint32_t batchCount = _L->GetDimsZ();

    const bool useBlock = (_L->GetDimsX() > 32 || _R->GetDimsX() > 32
                        || _L->GetDimsY() > 32 || _R->GetDimsY() > 32);

    const T* A = _L->DataRead();
    const T* B = _R->DataRead();
    T* C = _out->DataWrite();

    if (batchCount <= 1u)
    {
        if (useBlock)
        {
            if (pool)
                MulStridedBlock<Mode>(_out, _L, _R,
                                      L_rowStride, L_colStride,
                                      R_rowStride, R_colStride,
                                      outRowStride, outColStride,
                                      blockSize, numThreads, pool);
            else
                MulStridedBlock<Mode>(A, L_rows, L_cols, L_rowStride, L_colStride,
                                      B, R_rows, R_cols, R_rowStride, R_colStride,
                                      C, outRowStride, outColStride, blockSize);
        }
        else
        {
            MulStrided<Mode>(A, L_rows, L_cols, L_rowStride, L_colStride,
                             B, R_rows, R_cols, R_rowStride, R_colStride,
                             C, outRowStride, outColStride);
        }
        return;
    }

    const uint32_t L_planeStride = L_rowStride * L_rows;
    const uint32_t R_planeStride = R_rowStride * R_rows;
    const uint32_t outPlaneStride = outRowStride * outRows;

    auto batchWorker = [&](uint32_t z)
    {
        const T* Ab = A + z * L_planeStride;
        const T* Bb = B + z * R_planeStride;
        T* Cb = C + z * outPlaneStride;

        if (useBlock)
        {
            MulStridedBlock<Mode>(Ab, L_rows, L_cols, L_rowStride, L_colStride,
                                  Bb, R_rows, R_cols, R_rowStride, R_colStride,
                                  Cb, outRowStride, outColStride, blockSize);
        }
        else
        {
            MulStrided<Mode>(Ab, L_rows, L_cols, L_rowStride, L_colStride,
                             Bb, R_rows, R_cols, R_rowStride, R_colStride,
                             Cb, outRowStride, outColStride);
        }
    };

    if (pool)
    {
        std::latch latchCounter(batchCount);
        for (uint32_t z = 0; z < batchCount; ++z)
        {
            pool->Enqueue([&, z]
            {
                batchWorker(z);
                latchCounter.count_down();
            });
        }
        latchCounter.wait();
    }
    else
    {
        for (uint32_t z = 0; z < batchCount; ++z)
        {
            batchWorker(z);
        }
    }
}

