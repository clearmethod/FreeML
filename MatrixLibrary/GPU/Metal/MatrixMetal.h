#pragma once

#include <MatrixLibrary/MatrixBase.h>
#include <MatrixLibrary/GPU/Metal/MetalManager.h>

#include <cassert>
#include <cstring>
#include <vector>

#ifdef __APPLE__

template<class T>
class MatrixMetal : public MatrixBase<T>
{
    static_assert(std::is_same_v<T, float>, "T must be float");

    public:
    using MatrixBase<T>::GetValue;
    using MatrixBase<T>::SetValue;
    MatrixMetal(void* _data = nullptr)
    {
        MatrixBase<T>::m_hardwareName = "CPU";
        SetData(_data);
        m_gInstance = MetalManager::Instance();
    }

    MatrixMetal(T* _data, MatrixShape _dims,  std::string _name = "NoName")
    : MatrixMetal(_data)
    { 
        MatrixBase<T>::m_name = _name;
        MatrixBase<T>::m_dims = _dims;
    }

    ~MatrixMetal()
    {
        FreeMemory();
    }

    MatrixMetal(MatrixMetal&& other) noexcept
        : MatrixBase<T>(std::move(other)),
          m_dataHandles(other.m_dataHandles),
          m_derivedCBuffer(other.m_derivedCBuffer),
          m_offset(other.m_offset),
          m_derivedAll(other.m_derivedAll),
          m_gInstance(other.m_gInstance)
    {
        MatrixBase<T>::m_dims = other.m_dims;
        other.m_dataHandles={-1,-1};
        other.m_offset=0u;
        other.m_dims = MatrixShape();
    }

    virtual bool  AllocateMemory(uint32_t _elementCount) override
    {
        m_dataHandles.m_bufferHandle = m_gInstance->CreateBuffer(_elementCount * sizeof(T), _elementCount);
        m_offset = 0u;
        m_dataHandles.m_cbufferHandle = m_gInstance->AcquireCachedCBuffer(this->GetWidth(), this->GetHeight(), this->GetPlaneCount(), m_offset, 12345);

        SyncCBufferParams();
        assert(m_dataHandles.m_bufferHandle >= 0);
        assert(m_dataHandles.m_cbufferHandle >= 0);
        return true;
    }

    virtual void FreeMemory() override
    {
        if (!MetalManager::IsAlive() || !m_gInstance)
        {
            m_dataHandles = {-1, -1};
            return;
        }

        if (!m_derivedAll)
        {
            if (!m_derivedCBuffer && m_dataHandles.m_bufferHandle >= 0)
                m_gInstance->DeleteBuffer(m_dataHandles.m_bufferHandle);
            if (m_dataHandles.m_cbufferHandle >= 0)
                m_gInstance->ReleaseCachedCBuffer(m_dataHandles.m_cbufferHandle);
            m_dataHandles = {-1, -1};
        }
    }

    virtual void GetPlaneView(MatrixBase<T>* _matrixOut, uint32_t _zIndex) override
    {
        MatrixMetal<T>* matrixOutGPU = static_cast<MatrixMetal<T>*>(_matrixOut);
        matrixOutGPU->SetData(&m_dataHandles.m_bufferHandle);
        matrixOutGPU->SetShape(MatrixShape::Matrix(this->GetWidth(),this->GetHeight()));
        matrixOutGPU->m_derivedCBuffer = true;
        matrixOutGPU->m_derivedAll = false;
        matrixOutGPU->m_dataHandles.m_cbufferHandle = m_gInstance->AcquireCachedCBuffer(this->GetWidth(), this->GetHeight(), this->GetPlaneCount(), m_offset, 12345);
        matrixOutGPU->SetOffset(m_offset + (this->GetWidth() * this->GetHeight() * _zIndex));
    }

    void BindView(MatrixBase<T>* parent,const MatrixShape& shape,uint32_t relativeOffset) override
    {
        auto* source=static_cast<MatrixMetal<T>*>(parent);
        const auto offset=source->GetOffset()+relativeOffset;
        if(m_dataHandles.m_cbufferHandle<0 || m_derivedAll)
            m_dataHandles.m_cbufferHandle=m_gInstance->AcquireCachedCBuffer(shape.width,shape.height,
                shape.channels*shape.batch,offset,static_cast<uint32_t>(this->GetID()));
        m_dataHandles.m_bufferHandle=source->m_dataHandles.m_bufferHandle;
        m_offset=offset;m_derivedAll=false;m_derivedCBuffer=true;
        SetShape(shape);
    }

    virtual void SetOffset(const uint32_t _offset)
    {
        m_offset = _offset;
        m_gInstance->UpdateCachedCBuffer(m_dataHandles.m_cbufferHandle,
                                         this->GetWidth(),
                                         this->GetHeight(),
                                         this->GetPlaneCount(),
                                         m_offset,
                                         12345);
    }

    void SetShape(MatrixShape _dims) override
    {
        MatrixBase<T>::m_dims = _dims;
        SyncCBufferParams();
    }
    
    virtual void SetData(void* _ptr, uint32_t _offset = 0u) override
    {
        if (_ptr)
        {
            m_dataHandles = *(handles*)_ptr;
            m_derivedAll = true;
        }
        m_offset = _offset;
        if (_offset != 0)
        {
            m_derivedAll     = false;
            m_derivedCBuffer = true;
            m_dataHandles.m_cbufferHandle = m_gInstance->AcquireCachedCBuffer(this->GetWidth(), this->GetHeight(), this->GetPlaneCount(), m_offset, 12345);
            SetOffset(_offset);
        }
    }
    
    virtual void* GetData() override
    {
        return (void*)(&m_dataHandles);
    }

    virtual uint32_t GetOffset() override
    {
        return m_offset;
    }

    std::vector<T> GetRow(uint32_t _r) override
    {
        const T* start = DataRead() + (_r * MatrixBase<T>::GetWidth());
        return std::vector<T>(start, start + MatrixBase<T>::GetWidth());
    }

    std::vector<T> GetColumn(uint32_t _c) override
    {
        std::vector<T> col;
        col.reserve(MatrixBase<T>::GetHeight());
        for (uint32_t y = 0; y < MatrixBase<T>::GetHeight(); ++y)
        {
            col.push_back(DataRead()[y * MatrixBase<T>::GetWidth() + _c]);
        }
        return col;
    }

    inline void SetValue(uint32_t _x, uint32_t _y, T _val) override
    {
        assert(_x < MatrixBase<T>::GetWidth());
        assert(_y < MatrixBase<T>::GetHeight());
        DataWrite()[(_y * MatrixBase<T>::GetWidth()) + _x] = _val;
    }

    inline void SetValue(uint32_t _x, uint32_t _y, uint32_t _z, T _val) override
    {
        assert(_x < MatrixBase<T>::GetWidth());
        assert(_y < MatrixBase<T>::GetHeight());
        assert(_z < MatrixBase<T>::GetPlaneCount());
        DataWrite()[(MatrixBase<T>::GetWidth() * MatrixBase<T>::GetHeight() * _z) + (_y * MatrixBase<T>::GetWidth()) + _x] = _val;
    }

    inline T GetValue(uint32_t _x, uint32_t _y) override
    {
        assert(_x < MatrixBase<T>::GetWidth());
        assert(_y < MatrixBase<T>::GetHeight());
        return DataRead()[(_y * MatrixBase<T>::GetWidth()) + _x];
    }

    inline T GetValue(uint32_t _x, uint32_t _y, uint32_t _z) override
    {
        assert(_x < MatrixBase<T>::GetWidth());
        assert(_y < MatrixBase<T>::GetHeight());
        assert(_z < MatrixBase<T>::GetPlaneCount());
        return DataRead()[(MatrixBase<T>::GetWidth() * MatrixBase<T>::GetHeight() * _z) + (_y * MatrixBase<T>::GetWidth()) + _x];
    }

    inline T* DataWrite() override 
    { 
        auto* ptr = m_gInstance->GetMappedPointer(m_dataHandles.m_bufferHandle);
        if(!ptr)
        {
            //LOG_INFO() << "Mapping metal matrix " << MatrixBase<T>::GetName() << " to cpu.\n";
            MetalManager::Instance()->WaitForGPU();
            m_gInstance->CopyGPUToStaging(m_dataHandles.m_bufferHandle);
            m_gInstance->WaitForGPU();
            ptr = m_gInstance->MapBuffer(m_dataHandles.m_bufferHandle);
            assert(ptr);
            return (T*)ptr + m_offset;
        }
        else
        {
            m_gInstance->MarkMappingDirty(m_dataHandles.m_bufferHandle);
            return (T*)ptr + m_offset;
        }
    }

    inline const T* DataRead() override 
    { 
        auto* ptr = m_gInstance->GetMappedPointer(m_dataHandles.m_bufferHandle);
        if (!ptr)
        {
            //LOG_INFO() << "Mapping metal matrix " << MatrixBase<T>::GetName() << " to cpu.\n";
            MetalManager::Instance()->WaitForGPU();
            m_gInstance->CopyGPUToStaging(m_dataHandles.m_bufferHandle);
            m_gInstance->WaitForGPU();
            ptr = m_gInstance->MapBuffer(m_dataHandles.m_bufferHandle, false);
            assert(ptr);
            return (T*)ptr + m_offset;
        }
        else
        {
            return static_cast<const T*>(ptr) + m_offset;
        }
    }

    void GetRawData(std::vector<char>& _out) override
    {
        Unmap();
        const uint32_t count = GetElementCount();
        if (count == 0u)
        {
            _out.clear();
            return;
        }

        const T* src = DataRead();
        if (!src)
        {
            _out.clear();
            return;
        }

        const size_t bytes = static_cast<size_t>(count) * sizeof(T);
        _out.resize(bytes);
        std::memcpy(_out.data(), src, bytes);
        Unmap();
    }

    inline void Unmap()
    {
        auto* ptr = m_gInstance->GetMappedPointer(m_dataHandles.m_bufferHandle);
        if (!ptr)
            return;

        if (m_gInstance->IsMappingDirty(m_dataHandles.m_bufferHandle))
            m_gInstance->CopyStagingToGPU(m_dataHandles.m_bufferHandle);
        m_gInstance->UnmapBuffer(m_dataHandles.m_bufferHandle);
    }

    uint64_t GetOwnedBytes() const override
    {
        if(m_derivedAll || m_derivedCBuffer || !m_gInstance) return 0u;
        auto it=m_gInstance->m_bufferSizesMap.find(m_dataHandles.m_bufferHandle);
        return it==m_gInstance->m_bufferSizesMap.end()?0u:it->second;
    }

    uint32_t GetElementCount() const override
    {
        return this->GetShape().CheckedElementCount();
    }

    struct handles
    {
        int32_t  m_bufferHandle = -1;
        int32_t  m_cbufferHandle = -1;
    };

    handles  m_dataHandles      = {-1, -1};
    uint32_t m_offset           = 0;
    bool     m_derivedCBuffer   = false;
    bool     m_derivedAll       = false;

    MetalManager* m_gInstance;

    private:
    void SyncCBufferParams()
    {
        if (!m_gInstance)
        {
            return;
        }
        if (m_dataHandles.m_cbufferHandle < 0)
        {
            return;
        }
        m_gInstance->UpdateCachedCBuffer(m_dataHandles.m_cbufferHandle,
                                         this->GetWidth(),
                                         this->GetHeight(),
                                         this->GetPlaneCount(),
                                         m_offset,
                                         12345, this->GetChannelCount(), this->GetBatchCount());
    }


};



#endif
