
#pragma once

#include <ToolsLibrary/Tools.h>
#include "MatrixShape.h"

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <memory>

template<typename T>
class MatrixBase 
{
public:
    MatrixBase()
    {
        m_guid = m_guid = GenerateGuidUINT();
    }
    MatrixBase(MatrixBase&&) noexcept = default;
    MatrixBase& operator=(MatrixBase&&) noexcept = default;
    virtual                ~MatrixBase  () = default;
    virtual T*              DataWrite   () = 0;
    virtual const T*        DataRead    () = 0;

    uint64_t GetID()
    {
        return m_guid;
    }

    void SetID(uint64_t _id)
    {
        m_guid = _id;
    }

    virtual bool            AllocateMemory  (uint32_t _elementCount) = 0;
    virtual void            FreeMemory      () {}
    virtual uint64_t        GetOwnedBytes() const { return 0u; }

    virtual T               GetValue        (std::uint32_t x, std::uint32_t y)        = 0;
    virtual T               GetValue        (uint32_t _x, uint32_t _y, uint32_t _z)   = 0;
    virtual void            SetValue        (std::uint32_t x, std::uint32_t y, T v)         = 0;
    virtual void            SetValue        (uint32_t _x, uint32_t _y, uint32_t _z, T _val) = 0;

    
    virtual std::vector<T>  GetRow          (std::uint32_t r) = 0;
    virtual std::vector<T>  GetColumn       (std::uint32_t c) = 0;

    virtual std::uint32_t   GetElementCount() const = 0;

    virtual void GetRawData(std::vector<char>& _out) = 0;

    // Borrowed temporary plane; managed views use MatrixManager to retain storage.
    virtual void GetPlaneView(MatrixBase<T>* _matrixOut, uint32_t _zIndex) {};

    const MatrixShape& GetShape() const { return m_dims; }
    uint32_t GetBatchCount() const { return m_dims.batch; }
    uint32_t GetChannelCount() const { return m_dims.channels; }
    // Managed views retain the parent matrix without taking ownership of its buffer.
    void RetainViewParent(std::shared_ptr<void> parent,MatrixBase<T>* matrix)
    { m_viewParent=std::move(parent);m_viewParentMatrix=matrix; }
    MatrixBase<T>* GetViewParent() const { return m_viewParentMatrix; }
    virtual void BindView(MatrixBase<T>* parent,const MatrixShape& shape,uint32_t relativeOffset)
    {
        SetData(parent->GetData(),parent->GetOffset()+relativeOffset);
        SetShape(shape);
    }

    T GetValue(uint32_t b, uint32_t c, uint32_t x, uint32_t y)
    { return DataRead()[m_dims.Index(b,c,x,y)]; }
    void SetValue(uint32_t b, uint32_t c, uint32_t x, uint32_t y, T value)
    { DataWrite()[m_dims.Index(b,c,x,y)] = value; }

    virtual uint32_t GetWidth() const
    {
        return m_dims.width;
    }

    virtual uint32_t GetHeight() const
    {
        return m_dims.height;
    }

    virtual uint32_t GetPlaneCount() const
    {
        return MatrixShape::Matrix(m_dims.channels, m_dims.batch).CheckedElementCount();
    }

    virtual void SetShape(MatrixShape _dims)
    {
        MatrixBase<T>::m_dims = _dims;
    }

    virtual void      SetData(void* _ptr, uint32_t _offset = 0u) {};
	virtual void*     GetData()   { return nullptr;};
	virtual uint32_t  GetOffset() { return 0; };

    virtual std::string Description()
    {
        std::stringstream ss;
        ss << typeid(T).name() << "(" << GetWidth() << ", " << GetHeight() << ")";
        return ss.str();
    }

    std::string GetName()
    {
        return m_name;
    }
    void SetName(std::string _str)
    {
        m_name = _str;
    }

    std::string GetString(int _maxOut = 5)
    {
        std::stringstream ss;
        ss << "Name: " << m_name << "\n";
        ss << "Dims: " << m_dims.GetString() << "\n";
        int county = 0;
        for(uint32_t y = 0u; y < m_dims.height; y++)
        {
            county++;
            ss << "[ ";
            for(uint32_t x = 0u; x < m_dims.width; x++)
            {
                ss << std::fixed << std::setprecision(3) << GetValue(x,y);

                if(x != m_dims.width - 1)
                    ss << ",";
                
            }
            ss << "]\n";
            if (_maxOut > 0 && county >= _maxOut)
            {
                ss << "... \n";
                return ss.str();
            }
        }
        return ss.str();
    }

    std::string m_name           = "";
    std::string m_hardwareName   = "";
    MatrixShape      m_dims           = MatrixShape::Matrix(0u,0u);
    uint64_t    m_guid           = 0u;

private:
    std::shared_ptr<void> m_viewParent;
    MatrixBase<T>* m_viewParentMatrix=nullptr;

};
