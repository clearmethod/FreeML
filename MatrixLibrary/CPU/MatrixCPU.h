#pragma once

#include "../MatrixBase.h"

#include <cassert>
#include <cstring>

template<class T>
class MatrixCPU : public MatrixBase<T>
{
    public:
    using MatrixBase<T>::GetValue;
    using MatrixBase<T>::SetValue;
    MatrixCPU(void* _data = nullptr)
    {
        MatrixBase<T>::m_hardwareName = "CPU";
        SetData(_data);
    }

    MatrixCPU(T* _data, MatrixShape _dims,  std::string _name = "NoName")
    : MatrixCPU(_data)
    { 
        MatrixBase<T>::m_name = _name;
        MatrixBase<T>::m_dims = _dims;
    }

    ~MatrixCPU()
    {
    }

    MatrixCPU(MatrixCPU&& other) noexcept
        : MatrixBase<T>(std::move(other)),
          m_data(other.m_data)
    {
        MatrixBase<T>::m_dims = other.m_dims;
        m_ownedData = other.m_ownedData;
        m_ownedBytes = other.m_ownedBytes;
        other.m_ownedData = nullptr;
        other.m_ownedBytes = 0u;
        other.m_data = nullptr;
        other.m_dims = MatrixShape();
    }

    virtual bool  AllocateMemory(uint32_t _elementCount) override
    {
        if (m_ownedData) FreeMemory();
        if (uint64_t(_elementCount) * sizeof(T) > UINT32_MAX)
            throw std::overflow_error("Matrix allocation exceeds backend byte limits.");
        m_data = (T*)( ::operator new(size_t(_elementCount) * sizeof(T), std::align_val_t(64)));
        m_ownedData = m_data;
        m_ownedBytes = size_t(_elementCount)*sizeof(T);
        return m_data != nullptr;
    }

    virtual void FreeMemory() override
    {
        ::operator delete(m_ownedData, std::align_val_t(64));
        m_ownedData = nullptr;
        m_ownedBytes = 0u;
        m_data = nullptr;
    }

    virtual void GetPlaneView(MatrixBase<T>* _matrixOut, uint32_t _zIndex) override
    {
        MatrixCPU<T>* matrixOutCPU = static_cast<MatrixCPU<T>*>(_matrixOut);
        matrixOutCPU->SetData( &(((T*)m_data)[this->GetWidth() * this->GetHeight() * _zIndex]));
        matrixOutCPU->SetShape(MatrixShape::Matrix(this->GetWidth(),this->GetHeight()));
    }

    virtual void SetData(void* _ptr, uint32_t _offset = 0u) override
    {
		m_data = reinterpret_cast<T*>(_ptr) + _offset;
    }

    virtual void* GetData() override
    {
        return m_data;
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
        return m_data;
    }
    inline const T* DataRead() override 
    { 
        return m_data; 
    }

    void GetRawData(std::vector<char>& _out) override
    {
        const uint32_t count = GetElementCount();
        if (count == 0u || m_data == nullptr)
        {
            _out.clear();
            return;
        }

        const size_t bytes = static_cast<size_t>(count) * sizeof(T);
        _out.resize(bytes);
        std::memcpy(_out.data(), m_data, bytes);
    }

    uint32_t GetElementCount() const override
    {
        return this->GetShape().CheckedElementCount();
    }

    T* m_data = nullptr;
    uint64_t GetOwnedBytes() const override { return m_ownedBytes; }
private:
    T* m_ownedData = nullptr;
    uint64_t m_ownedBytes = 0u;
};
