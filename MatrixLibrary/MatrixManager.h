#pragma once

#include <ToolsLibrary/Logger.h>

#include <cstdint>
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <functional>
#include <cassert>
#include <utility>

#include "MatrixBase.h"

typedef int64_t MatrixHandle;

constexpr MatrixHandle InvalidMatrixHandle = -1;


template<typename T, class Mat>
class Layer;

template <class T, class MatType>
class MatrixManager
{
    public:
    using ScratchKey = MatrixShape;
    struct MatrixRef
    {
        MatrixRef() = default;
        MatrixRef(MatrixManager* _mgr, MatrixHandle _handle, MatType* _ptr)
            : mgr(_mgr), handle(_handle), ptr(_ptr)
        {
        }

        MatrixRef(const MatrixRef& other)
            : mgr(other.mgr), handle(other.handle), ptr(other.ptr)
        {
            AddRef();
        }

        MatrixRef& operator=(const MatrixRef& other)
        {
            if (this != &other)
            {
                Reset();
                mgr = other.mgr;
                handle = other.handle;
                ptr = other.ptr;
                AddRef();
            }
            return *this;
        }

        MatrixRef(MatrixRef&& other) noexcept
        {
            mgr = other.mgr;
            handle = other.handle;
            ptr = other.ptr;
            other.mgr = nullptr;
            other.handle = InvalidMatrixHandle;
            other.ptr = nullptr;
        }

        MatrixRef& operator=(MatrixRef&& other) noexcept
        {
            if (this != &other)
            {
                Reset();
                mgr = other.mgr;
                handle = other.handle;
                ptr = other.ptr;
                other.mgr = nullptr;
                other.handle = InvalidMatrixHandle;
                other.ptr = nullptr;
            }
            return *this;
        }

        ~MatrixRef()
        {
            Reset();
        }

        void Reset()
        {
            if (mgr && handle != InvalidMatrixHandle)
            {
                mgr->DecRef(handle);
            }
            mgr = nullptr;
            handle = InvalidMatrixHandle;
            ptr = nullptr;
        }

        MatType* get() const { return ptr; }
        MatrixHandle getHandle() const { return handle; }
        MatType* operator->() const { return ptr; }
        explicit operator bool() const { return ptr != nullptr; }
        bool operator==(const MatrixRef& other) const { return handle == other.handle; }
        bool operator!=(const MatrixRef& other) const { return handle != other.handle; }

        private:
        void AddRef()
        {
            if (mgr && handle != InvalidMatrixHandle)
            {
                mgr->IncRef(handle);
            }
        }

        MatrixManager* mgr = nullptr;
        MatrixHandle handle = InvalidMatrixHandle;
        MatType* ptr = nullptr;
    };

    static MatrixManager& Instance()
    {
        static MatrixManager instance;
        return instance;
    }

    MatrixManager(const MatrixManager&) = delete;
    MatrixManager& operator=(const MatrixManager&) = delete;
    MatrixManager(MatrixManager&&) = delete;
    MatrixManager& operator=(MatrixManager&&) = delete;

    std::string GetString(bool _all = false)
    {
        std::stringstream ss;
        ss << "MatrixManager\n";
        ss << "Total matrices: " << m_HandleToMatrix.size() << "\n";
        ss << "Scratch cached: " << m_ScratchRefs.size() << "\n";
        ss << "Skeleton cached: " << m_SkeletonStack.size() << "\n";

        uint64_t totalBytes = 0u;
        for (const auto& kv : m_HandleToMatrix)
        {
            const MatrixHandle handle = kv.first;
            MatrixBase<T>* mat = kv.second;
            if (!mat)
            {
                continue;
            }

            const uint64_t elements = static_cast<uint64_t>(mat->GetElementCount());
            const uint64_t bytes = elements * sizeof(T);
            const uint64_t ownedBytes=mat->GetOwnedBytes();
            totalBytes += ownedBytes;

            uint32_t refCount = 0u;
            auto refIt = m_HandleRefCount.find(handle);
            if (refIt != m_HandleRefCount.end())
            {
                refCount = refIt->second;
            }

            if(_all)
            {
                ss << "[" << handle << "] "
                   << mat->GetName()
                   << " dims " << mat->GetShape().GetString()
                   << " elements " << elements
                   << " bytes " << bytes
                   << " owned bytes " << ownedBytes
                   << " refs " << refCount
                   << "\n";
            }
        }

        const double totalMB = static_cast<double>(totalBytes) / (1024.0 * 1024.0);
        const double totalGB = static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0);

        ss << "Total allocated: "
           << totalBytes << " bytes, "
           << std::fixed << std::setprecision(2)
           << totalMB << " MB, "
           << totalGB << " GB\n";

        return ss.str();
    }

    uint64_t GetAllocatedBytes() const
    {
        uint64_t bytes=0u;
        for(const auto& [handle,matrix] : m_HandleToMatrix)
            if(matrix) bytes+=matrix->GetOwnedBytes();
        return bytes;
    }

    MatrixRef AllocateMatrix(const MatrixShape& _dims, const std::string& _name, Layer<T, MatType>* _owningLayer = nullptr)
    {
        if (!_dims.ElementCount())
            throw std::invalid_argument("Cannot allocate an empty matrix shape.");
        const uint32_t count = _dims.CheckedElementCount(UINT32_MAX / sizeof(T));
        MatType* newMat = new MatType(nullptr, _dims, _name);
        newMat->AllocateMemory(count);
        MatrixHandle handle = AddMatrix(newMat, _owningLayer);
        return Acquire(handle);
    }

    MatrixRef CreateView(const MatrixRef& parent, const MatrixShape& shape,
                         uint32_t relativeOffset = 0u, const std::string& name = "View")
    {
        if (!parent || GetHandle(parent.get()) == InvalidMatrixHandle)
            throw std::invalid_argument("A view requires a matrix owned by this manager.");
        const uint32_t count = shape.CheckedElementCount(UINT32_MAX / sizeof(T));
        if (!count || uint64_t(relativeOffset) + count > parent->GetElementCount())
            throw std::out_of_range("Matrix view exceeds its parent extent.");
        const uint64_t offset = uint64_t(parent->GetOffset()) + relativeOffset;
        if (offset > UINT32_MAX) throw std::overflow_error("Matrix view offset exceeds backend limits.");
        auto view = std::make_unique<MatType>();
        view->BindView(parent.get(),shape,relativeOffset);
        view->SetName(name);
        view->RetainViewParent(std::make_shared<MatrixRef>(parent),parent.get());
        const MatrixHandle handle = AddMatrix(view.get());
        view.release();
        return Acquire(handle);
    }

    void UpdateView(MatrixRef& view,const MatrixRef& parent,const MatrixShape& shape,uint32_t relativeOffset=0u)
    {
        if(!view) { view=CreateView(parent,shape,relativeOffset);return; }
        if(!parent || view.get()==parent.get() || !view->GetViewParent()
            || GetHandle(parent.get())==InvalidMatrixHandle)
            throw std::invalid_argument("Only managed views can be rebound to a managed parent.");
        for(auto* ancestor=parent.get();ancestor;ancestor=static_cast<MatType*>(ancestor->GetViewParent()))
            if(ancestor==view.get()) throw std::invalid_argument("Matrix views cannot form ownership cycles.");
        const auto count=shape.CheckedElementCount(UINT32_MAX/sizeof(T));
        if(!count || uint64_t(relativeOffset)+count>parent->GetElementCount()
            || uint64_t(parent->GetOffset())+relativeOffset>UINT32_MAX)
            throw std::out_of_range("Rebound view exceeds its parent extent.");
        view->BindView(parent.get(),shape,relativeOffset);
        if(view->GetViewParent()!=parent.get())
            view->RetainViewParent(std::make_shared<MatrixRef>(parent),parent.get());
    }

    MatrixRef ReshapeView(const MatrixRef& parent, const MatrixShape& shape)
    {
        if (!parent || shape.ElementCount() != parent->GetElementCount())
            throw std::invalid_argument("Reshape must preserve the matrix element count.");
        return CreateView(parent, shape, 0u, "ReshapeView");
    }

    MatrixRef BatchView(const MatrixRef& parent, uint32_t batch)
    {
        if (!parent || batch >= parent->GetBatchCount())
            throw std::out_of_range("Batch index exceeds the matrix extent.");
        MatrixShape shape = parent->GetShape();
        shape.batch = 1u;
        return CreateView(parent, shape, batch * shape.CheckedElementCount(), "BatchView");
    }

    MatrixRef ChannelView(const MatrixRef& parent, uint32_t batch, uint32_t channel)
    {
        if (!parent || batch >= parent->GetBatchCount() || channel >= parent->GetChannelCount())
            throw std::out_of_range("Channel index exceeds the matrix extent.");
        const auto& shape = parent->GetShape();
        const auto plane = MatrixShape::Matrix(shape.width, shape.height);
        return CreateView(parent, plane,
            (batch * shape.channels + channel) * plane.CheckedElementCount(), "ChannelView");
    }

    MatrixHandle AddMatrix(MatType* _matrix, Layer<T, MatType>* _owningLayer = nullptr)
    {
        if (_matrix == nullptr)
        {
            return InvalidMatrixHandle;
        }

        MatrixHandle handle = m_nextHandle++;
        m_HandleToMatrix.emplace(handle, _matrix);
        m_MatrixToHandle[_matrix] = handle;
        m_HandleRefCount[handle] = 0u;
        return handle;
    }

    MatType* GetScratch(const MatrixShape& _dims)
    {
        ScratchKey key = _dims;
        auto it = m_ScratchRefs.find(key);
        if (it != m_ScratchRefs.end())
        {
            MatType* existing = it->second.get();
            if (existing && existing->GetShape() == _dims)
            {
                return existing;
            }
            m_ScratchRefs.erase(it);
        }

        MatrixRef scratch = AllocateMatrix(_dims, "Scratch");
        if (scratch.get())
        {
            m_ScratchRefs[key] = std::move(scratch);
            return m_ScratchRefs[key].get();
        }
        return nullptr;
    }

    bool IsSkeleton(MatType* _mat)
    {
        if (!_mat)
        {
            return false;
        }
        const MatrixHandle handle = GetHandle(_mat);
        if (handle == InvalidMatrixHandle)
        {
            return false;
        }
        return m_SkeletonHandles.find(handle) != m_SkeletonHandles.end();
    }

    MatrixRef GetSkeleton(const std::string& _name = "Skeleton")
    {
        if (!m_SkeletonStack.empty())
        {
            MatrixRef ref = std::move(m_SkeletonStack.back());
            m_SkeletonStack.pop_back();
            m_SkeletonHandles[ref.getHandle()] = true;
            return ref;
        }

        MatType* skeleton = new MatType(nullptr, MatrixShape(), _name);
        AddMatrix(skeleton);
        MatrixHandle handle = GetHandle(skeleton);
        if (handle != InvalidMatrixHandle)
        {
            m_SkeletonHandles[handle] = true;
        }
        return Acquire(handle);
    }

    void ReleaseSkeleton(MatrixRef& _ref)
    {
        if (_ref.getHandle() == InvalidMatrixHandle)
        {
            return;
        }
        MatType* mat = GetMatrix(_ref.getHandle());
        if (!mat)
        {
            return;
        }
        m_SkeletonHandles[_ref.getHandle()] = true;
        m_SkeletonStack.push_back(std::move(_ref));
    }

    void ReleaseSkeleton(MatrixHandle _handle)
    {
        if (_handle == InvalidMatrixHandle)
        {
            return;
        }
        MatType* mat = GetMatrix(_handle);
        if (!mat)
        {
            return;
        }
        m_SkeletonHandles[_handle] = true;
        MatrixRef ref = Acquire(_handle);
        if (ref.get())
        {
            m_SkeletonStack.push_back(std::move(ref));
        }
    }

    void ReleaseSkeleton(MatType* _matrix)
    {
        if (!_matrix)
        {
            return;
        }
        ReleaseSkeleton(GetHandle(_matrix));
    }

    MatType* GetMatrix(MatrixHandle _handle) const
    {
        auto it = m_HandleToMatrix.find(_handle);
        if (it == m_HandleToMatrix.end())
        {
            return nullptr;
        }

        return dynamic_cast<MatType*>(it->second);
    }

    MatType* GetMatrixByGuid(uint64_t guid) const
    {
        for (const auto& entry : m_HandleToMatrix)
        {
            MatrixBase<T>* mat = entry.second;
            if (!mat)
            {
                continue;
            }
            if (mat->GetID() == guid)
            {
                return dynamic_cast<MatType*>(mat);
            }
        }
        return nullptr;
    }

    MatrixHandle GetHandle(MatType* _matrix) const
    {
        auto it = m_MatrixToHandle.find(_matrix);
        if (it == m_MatrixToHandle.end())
        {
            return InvalidMatrixHandle;
        }

        return it->second;
    }

    MatrixRef Acquire(MatType* _matrix)
    {
        if (!_matrix)
        {
            return MatrixRef();
        }
        return Acquire(GetHandle(_matrix));
    }

    MatrixRef Acquire(MatrixHandle _handle)
    {
        auto it = m_HandleToMatrix.find(_handle);
        if (it == m_HandleToMatrix.end())
        {
            return MatrixRef();
        }

        IncRef(_handle);
        return MatrixRef(this, _handle, dynamic_cast<MatType*>(it->second));
    }

    void RemoveMatrix(MatrixHandle _handle)
    {
        if (_handle == InvalidMatrixHandle)
        {
            return;
        }
        RemoveScratchHandle(_handle);
        RemoveSkeletonHandle(_handle);
        TryDeleteNow(_handle);
    }

    void RemoveMatrix(MatType* _matrix)
    {
        if (!_matrix)
        {
            return;
        }
        RemoveMatrix(GetHandle(_matrix));
    }

    void RemoveMatrix(MatrixRef& _ref)
    {
        if (!_ref)
        {
            return;
        }
        MatrixHandle handle = _ref.getHandle();
        _ref.Reset();
        RemoveMatrix(handle);
    }

    private:
    MatrixManager() = default;
    ~MatrixManager() 
    {
        LOG_INFO() << "Shutting down MatrixManager";
        // Release MatrixRef holders while core maps are still valid.
        m_ScratchRefs.clear();
        m_SkeletonStack.clear();
    }

    // Core state used by MatrixRef callbacks during shutdown.
    std::map<MatrixHandle, MatType*>            m_HandleToMatrix;
    std::map<MatType*, MatrixHandle>            m_MatrixToHandle;
    std::map<MatrixHandle, Layer<T, MatType>*>  m_HandleToOwner;
    std::map<MatrixHandle, uint32_t>            m_HandleRefCount;
    std::map<MatrixHandle, bool>                m_SkeletonHandles;

    // MatrixRef holders must destruct before core maps.
    std::map<ScratchKey, MatrixRef>             m_ScratchRefs;
    std::vector<MatrixRef>                      m_SkeletonStack;



    MatrixHandle                                m_nextHandle = 0;

    void IncRef(MatrixHandle _handle)
    {
        auto it = m_HandleRefCount.find(_handle);
        if (it != m_HandleRefCount.end())
        {
            ++it->second;
        }
    }

    void DecRef(MatrixHandle _handle)
    {
        auto it = m_HandleRefCount.find(_handle);
        if (it == m_HandleRefCount.end() || it->second == 0u)
        {
            return;
        }
        --it->second;
        if (it->second == 0u)
        {
            TryDeleteNow(_handle);
        }
    }

    void TryDeleteNow(MatrixHandle _handle)
    {
        auto it = m_HandleToMatrix.find(_handle);
        if (it == m_HandleToMatrix.end())
        {
            return;
        }

        const uint32_t refCount = m_HandleRefCount[_handle];
        if (refCount != 0u)
        {
            return;
        }

        MatType* mat = it->second;
        if (const auto ownedBytes=mat->GetOwnedBytes(); ownedBytes>0u)
            LOG_INFO() << "Freeing: " << mat->GetName() << ". Size: " << ownedBytes << " bytes.";
        else
            LOG_DEBUG() << "Releasing matrix metadata: " << mat->GetName();
        if(!IsSkeleton(mat))
            mat->FreeMemory();
        delete mat;

        m_MatrixToHandle.erase(mat);
        m_HandleToMatrix.erase(it);
        m_HandleRefCount.erase(_handle);
        m_SkeletonHandles.erase(_handle);
    }

    void RemoveScratchHandle(MatrixHandle _handle)
    {
        for (auto it = m_ScratchRefs.begin(); it != m_ScratchRefs.end(); )
        {
            if (it->second.getHandle() == _handle)
            {
                it = m_ScratchRefs.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    void RemoveSkeletonHandle(MatrixHandle _handle)
    {
        for (auto it = m_SkeletonStack.begin(); it != m_SkeletonStack.end(); )
        {
            if (it->getHandle() == _handle)
            {
                it = m_SkeletonStack.erase(it);
            }
            else
            {
                ++it;
            }
        }
        // Keep the borrowed-storage marker until TryDeleteNow deletes the
        // matrix. Live references must never turn a view into an owner.
    }
};



