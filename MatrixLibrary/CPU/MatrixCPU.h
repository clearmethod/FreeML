#pragma once

#include "../MatrixBase.h"

#include <cassert>
#include <cstring>

template<class T>
class MatrixCPU : public MatrixBase<T>
{
    public:
    MatrixCPU(void* _data = nullptr)
    {
        MatrixBase<T>::m_hardwareName = "CPU";
        SetData(_data);
    }

    MatrixCPU(T* _data, Dims3D _dims,  std::string _name = "NoName")
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
        other.m_data = nullptr;
        other.m_dims = Dims3D();
    }

    virtual bool  AllocateMemory(uint32_t _elementCount) override
    {
        m_data = (T*)( ::operator new(_elementCount * sizeof(T), std::align_val_t(64)));
        return m_data != nullptr;
    }

    virtual void FreeMemory() override
    {
        ::operator delete(DataWrite(), std::align_val_t(64));
    }

    virtual void GetSliceZ(MatrixBase<T>* _matrixOut, uint32_t _zIndex) override
    {
        MatrixCPU<T>* matrixOutCPU = static_cast<MatrixCPU<T>*>(_matrixOut);
        matrixOutCPU->SetData( &(((T*)m_data)[this->GetDimsX() * this->GetDimsY() * _zIndex]));
        matrixOutCPU->SetDims({this->GetDimsX(), this->GetDimsY(), 1});
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
        return MatrixBase<T>::GetDimsX() * MatrixBase<T>::GetDimsY() * MatrixBase<T>::GetDimsZ();
    }

    T* m_data = nullptr;
};
