
#pragma once

#include "MatrixBase.h"

template<class T>
class MatrixDynamicUnset : public MatrixBase<T>
{

    virtual bool  AllocateMemory(uint32_t _elementCount) override
    {
        return true;
    }

    std::vector<T> GetRow(uint32_t _r) override
    {
        return {};
    }

    std::vector<T> GetColumn(uint32_t _c) override
    {
        return {};
    }

    inline void SetValue(uint32_t _x, uint32_t _y, T _val) override
    {
    }

    inline void SetValue(uint32_t _x, uint32_t _y, uint32_t _z, T _val) override
    {
    }

    inline T GetValue(uint32_t _x, uint32_t _y) override
    {
        return 0;
    }

    inline T GetValue(uint32_t _x, uint32_t _y, uint32_t _z) override
    {
        return 0;
    }

    inline T* DataWrite() override 
    { 
        return nullptr;
    }
    inline const T* DataRead() override 
    { 
        return nullptr; 
    }

    void GetRawData(std::vector<char>& _out) override
    {
        _out.clear();
    }

    uint32_t GetElementCount() const override
    {
        return 0;
    }
};
