
#pragma once

#include <ToolsLibrary/Tools.h>

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

struct Dims3D
{
    Dims3D() = default;
    Dims3D(uint32_t _x, uint32_t _y = 1u, uint32_t _z = 1u) : x(_x), y(_y), z(_z){}
    Dims3D(const Dims3D& _other) = default;
    Dims3D(Dims3D&& _other) noexcept = default;
    Dims3D& operator=(const Dims3D& _other) = default;
    Dims3D& operator=(Dims3D&& _other) noexcept = default;
    uint32_t x = 0u;
    uint32_t y = 0u;
    uint32_t z = 0u;

    std::string GetString() const
    {
        return "(" + std::to_string(x) + ", " 
                   + (y != 1 ? std::to_string(y) : "") 
                   + (z != 1 ? ", " + std::to_string(z) : "")
                   + ")";
    }
};

template<typename T>
class MatrixBase 
{
public:
    MatrixBase()
    {
        m_guid = m_guid = GenerateGuidUINT();
    }
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

    virtual T               GetValue        (std::uint32_t x, std::uint32_t y)        = 0;
    virtual T               GetValue        (uint32_t _x, uint32_t _y, uint32_t _z)   = 0;
    virtual void            SetValue        (std::uint32_t x, std::uint32_t y, T v)         = 0;
    virtual void            SetValue        (uint32_t _x, uint32_t _y, uint32_t _z, T _val) = 0;

    
    virtual std::vector<T>  GetRow          (std::uint32_t r) = 0;
    virtual std::vector<T>  GetColumn       (std::uint32_t c) = 0;

    virtual std::uint32_t   GetElementCount() const = 0;

    virtual void GetRawData(std::vector<char>& _out) = 0;

    virtual void GetSliceX(MatrixBase<T>* _matrixOut, uint32_t _zIndex) {};
    virtual void GetSliceY(MatrixBase<T>* _matrixOut, uint32_t _zIndex) {};
    virtual void GetSliceZ(MatrixBase<T>* _matrixOut, uint32_t _zIndex) {};

    virtual uint32_t GetDimsX() const
    {
        return m_dims.x;
    }

    virtual uint32_t GetDimsY() const
    {
        return m_dims.y;
    }

    virtual uint32_t GetDimsZ() const
    {
        return m_dims.z;
    }

    virtual Dims3D& GetDims()
    {
        return m_dims;
    }

    virtual void SetDims(uint32_t _x, uint32_t _y = 1u, uint32_t _z = 1u)
    {
        MatrixBase<T>::m_dims.x = _x;
        MatrixBase<T>::m_dims.y = _y;
        MatrixBase<T>::m_dims.z = _z;
    }

    virtual void SetDims(Dims3D _dims)
    {
        MatrixBase<T>::m_dims = _dims;
    }

    virtual void      SetData(void* _ptr, uint32_t _offset = 0u) {};
	virtual void*     GetData()   { return nullptr;};
	virtual uint32_t  GetOffset() { return 0; };

    virtual std::string Description()
    {
        std::stringstream ss;
        ss << typeid(T).name() << "(" << GetDimsX() << ", " << GetDimsY() << ")";
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
        for(uint32_t y = 0u; y < m_dims.y; y++)
        {
            county++;
            ss << "[ ";
            for(uint32_t x = 0u; x < m_dims.x; x++)
            {
                ss << std::fixed << std::setprecision(3) << GetValue(x,y);

                if(x != m_dims.x - 1)
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
    Dims3D      m_dims           = Dims3D(0u, 0u);
    uint64_t    m_guid           = 0u;

};
