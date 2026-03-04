#pragma once

#include <MatrixLibrary/MatrixBase.h>
#include <MatrixLibrary/GPU/DirectX11/DirectX11Manager.h>

#include <cassert>
#include <cstring>
#include <vector>

#ifdef _WIN32

template<class T>
class MatrixDX11 : public MatrixBase<T>
{
    static_assert(std::is_same_v<T, float>, "T must be float");

    public:
    MatrixDX11(void* _data = nullptr)
    {
        MatrixBase<T>::m_hardwareName = "CPU";
        SetData(_data);
        m_gInstance = DirectX11Manager::Instance();
    }

    MatrixDX11(T* _data, Dims3D _dims,  std::string _name = "NoName")
    : MatrixDX11(_data)
    { 
        MatrixBase<T>::m_name = _name;
        MatrixBase<T>::m_dims = _dims;
    }

    ~MatrixDX11()
    {
        FreeMemory();
    }

    MatrixDX11(MatrixDX11&& other) noexcept
        : MatrixBase<T>(std::move(other)),
          m_dataHandles(other.m_dataHandles),
          m_derivedCBuffer(other.m_derivedCBuffer),
          m_offset(other.m_offset)
    {
        MatrixBase<T>::m_dims = other.m_dims;
        other.m_data = nullptr;
        other.m_dims = Dims3D();
    }

    virtual bool  AllocateMemory(uint32_t _elementCount) override
    {
        m_dataHandles.m_bufferHandle = m_gInstance->CreateBuffer(_elementCount * sizeof(T), _elementCount);
        m_offset = 0u;
        m_dataHandles.m_cbufferHandle = m_gInstance->AcquireCachedCBuffer(this->GetDimsX(), this->GetDimsY(), this->GetDimsZ(), m_offset, 12345);

        assert(m_dataHandles.m_bufferHandle >= 0);
        assert(m_dataHandles.m_cbufferHandle >= 0);
        return true;
    }

    virtual void FreeMemory() override
    {
        if (!DirectX11Manager::IsAlive() || !m_gInstance)
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

    virtual void GetSliceZ(MatrixBase<T>* _matrixOut, uint32_t _zIndex) override
    {
        MatrixDX11<T>* matrixOutGPU = static_cast<MatrixDX11<T>*>(_matrixOut);
        matrixOutGPU->SetData(&m_dataHandles.m_bufferHandle);
        matrixOutGPU->SetDims({this->GetDimsX(), this->GetDimsY(), 1});
        matrixOutGPU->m_derivedCBuffer = true;
        matrixOutGPU->m_derivedAll = false;
        matrixOutGPU->m_dataHandles.m_cbufferHandle = m_gInstance->AcquireCachedCBuffer(this->GetDimsX(), this->GetDimsY(), this->GetDimsZ(), m_offset, 12345);
        matrixOutGPU->SetOffset(m_offset + (this->GetDimsX() * this->GetDimsY() * _zIndex));

    }

    virtual void SetOffset(const uint32_t _offset)
    {
        m_offset = _offset;
        m_gInstance->UpdateCachedCBuffer(m_dataHandles.m_cbufferHandle,
                                         this->GetDimsX(),
                                         this->GetDimsY(),
                                         this->GetDimsZ(),
                                         m_offset,
                                         12345);
    }

    void SetDims(uint32_t _x, uint32_t _y = 1u, uint32_t _z = 1u) override
    {
        MatrixBase<T>::m_dims.x = _x;
        MatrixBase<T>::m_dims.y = _y;
        MatrixBase<T>::m_dims.z = _z;
        SyncCBufferParams();
    }

    void SetDims(Dims3D _dims) override
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
        if (_offset != 0)
        {
            m_derivedAll     = false;
            m_derivedCBuffer = true;
            m_dataHandles.m_cbufferHandle = m_gInstance->AcquireCachedCBuffer(this->GetDimsX(), this->GetDimsY(), this->GetDimsZ(), m_offset, 12345);
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
        const T* start = DataRead() + (_r * MatrixBase<T>::GetDimsX());
        return std::vector<T>(start, start + MatrixBase<T>::GetDimsX());
    }

    std::vector<T> GetColumn(uint32_t _c) override
    {
        std::vector<T> col;
        col.reserve(MatrixBase<T>::GetDimsY());
        for (uint32_t y = 0; y < MatrixBase<T>::GetDimsY(); ++y)
        {
            col.push_back(DataRead()[y * MatrixBase<T>::GetDimsX() + _c]);
        }
        return col;
    }

    inline void SetValue(uint32_t _x, uint32_t _y, T _val) override
    {
        assert(_x < MatrixBase<T>::GetDimsX());
        assert(_y < MatrixBase<T>::GetDimsY());
        DataWrite()[(_y * MatrixBase<T>::GetDimsX()) + _x] = _val;
    }

    inline void SetValue(uint32_t _x, uint32_t _y, uint32_t _z, T _val) override
    {
        assert(_x < MatrixBase<T>::GetDimsX());
        assert(_y < MatrixBase<T>::GetDimsY());
        assert(_z < MatrixBase<T>::GetDimsZ());
        DataWrite()[(MatrixBase<T>::GetDimsX() * MatrixBase<T>::GetDimsY() * _z) + (_y * MatrixBase<T>::GetDimsX()) + _x] = _val;
    }

    inline T GetValue(uint32_t _x, uint32_t _y) override
    {
        assert(_x < MatrixBase<T>::GetDimsX());
        assert(_y < MatrixBase<T>::GetDimsY());
        return DataRead()[(_y * MatrixBase<T>::GetDimsX()) + _x];
    }

    
    inline T GetValue(uint32_t _x, uint32_t _y, uint32_t _z) override
    {
        assert(_x < MatrixBase<T>::GetDimsX());
        assert(_y < MatrixBase<T>::GetDimsY());
        assert(_z < MatrixBase<T>::GetDimsZ());
        return DataRead()[(MatrixBase<T>::GetDimsX() * MatrixBase<T>::GetDimsY() * _z) + (_y * MatrixBase<T>::GetDimsX()) + _x];
    }

    inline T* DataWrite() override 
    { 
        auto* ptr = m_gInstance->GetMappedPointer(m_dataHandles.m_bufferHandle);
        if(!ptr)
        {
            DirectX11Manager::Instance()->WaitForGPU();
            m_gInstance->CopyGPUToStaging(m_dataHandles.m_bufferHandle);
            m_gInstance->WaitForGPU();
            D3D11_MAPPED_SUBRESOURCE mapped = {};
            ptr = m_gInstance->MapBuffer(m_dataHandles.m_bufferHandle, D3D11_MAP_READ_WRITE);
            assert(ptr);
            return (T*)ptr + m_offset;
        }
        else
        {
            return (T*)ptr + m_offset;
        }
    }

    inline const T* DataRead() override 
    { 
        auto* ptr = m_gInstance->GetMappedPointer(m_dataHandles.m_bufferHandle);
        if (!ptr)
        {
            DirectX11Manager::Instance()->WaitForGPU();
            m_gInstance->CopyGPUToStaging(m_dataHandles.m_bufferHandle);
            m_gInstance->WaitForGPU();
            D3D11_MAPPED_SUBRESOURCE mapped = {};
            ptr = m_gInstance->MapBuffer(m_dataHandles.m_bufferHandle, D3D11_MAP_READ_WRITE);
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

        m_gInstance->CopyStagingToGPU(m_dataHandles.m_bufferHandle);
        m_gInstance->UnmapBuffer(m_dataHandles.m_bufferHandle);
    }

    uint32_t GetElementCount() const override
    {
        return MatrixBase<T>::GetDimsX() * MatrixBase<T>::GetDimsY() * MatrixBase<T>::GetDimsZ();
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

    DirectX11Manager* m_gInstance;

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
                                         this->GetDimsX(),
                                         this->GetDimsY(),
                                         this->GetDimsZ(),
                                         m_offset,
                                         12345);
    }


};



#endif
