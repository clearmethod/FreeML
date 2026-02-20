#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include <MatrixLibrary/MatrixManager.h>

template< class T, class Mat>
class Layer;


template <class T, class Mat>
class Datablob
{
public:
    using MatrixRef = typename MatrixManager<T, Mat>::MatrixRef;
    std::map<std::string, MatrixRef>        m_matrixRefData;
    std::map<std::string, int>              m_intData;
    std::map<std::string, uint32_t>         m_uintData;
    std::map<std::string, float>            m_floatData;
    std::map<std::string, Datablob<T,Mat>*> m_ChildBlobData;
    std::map<std::string, Layer<T, Mat>*>   m_LayerData;

    ~Datablob()
    {
        for (auto& [_, layer] : m_LayerData)
        {
            if(layer)
                delete layer;
        }

        for (auto& [_, blob] : m_ChildBlobData)
        {
            if (blob)
                delete blob;
        }
    }

    const std::map<std::string, MatrixRef>& GetAllMatrixRefData() const
    {
        return m_matrixRefData;
    }

    const std::map<std::string, int>& GetAllIntData() const
    {
        return m_intData;
    }

    const std::map<std::string, uint32_t>& GetAllUIntData() const
    {
        return m_uintData;
    }

    const std::map<std::string, float>& GetAllFloatData() const
    {
        return m_floatData;
    }

    Layer<T, Mat>* GetLayer(const std::string& _name)
    {
        auto res = m_LayerData.find(_name);
        if(res == m_LayerData.end())
        {
            //LOG_INFO() << "Failed to find data on key: " << _name;
            return nullptr;
        }
        else
            return res->second;
    }

    Datablob<T,Mat>* GetBlob(const std::string& _name)
    {
        auto res = m_ChildBlobData.find(_name);
        if(res == m_ChildBlobData.end())
        {
            //LOG_INFO() << "Failed to find data on key: " << _name;
            return nullptr;
        }
        else
            return res->second;
    }

    template<class MatOut>
    MatOut* GetMatrix(const std::string& _name)
    {
        auto res = m_matrixRefData.find(_name);
        if (res == m_matrixRefData.end() || !res->second.get())
        {
            //LOG_INFO() << "Failed to find data on key: " << _name;
            return (MatOut*)nullptr;
        }
        return dynamic_cast<MatOut*>(res->second.get());
    }

    // Helper overload to avoid dependent-template syntax at call sites.
    Mat* GetMatrix(const std::string& _name)
    {
        return GetMatrix<Mat>(_name);
    }

    const Mat* GetMatrix(const std::string& _name) const
    {
        auto res = m_matrixRefData.find(_name);
        if (res == m_matrixRefData.end() || !res->second.get())
        {
            return nullptr;
        }
        return dynamic_cast<const Mat*>(res->second.get());
    }

    MatrixHandle GetMatrixHandle(const std::string& _name) const
    {
        auto res = m_matrixRefData.find(_name);
        if (res == m_matrixRefData.end() || !res->second.get())
        {
            return InvalidMatrixHandle;
        }
        return MatrixManager<T, Mat>::Instance().GetHandle(res->second.get());
    }

    MatrixRef AcquireMatrix(const std::string& _name)
    {
        MatrixHandle handle = GetMatrixHandle(_name);
        if (handle == InvalidMatrixHandle)
        {
            return MatrixRef();
        }
        return MatrixManager<T, Mat>::Instance().Acquire(handle);
    }

    int GetInt(const std::string& _name)
    {
        auto res = m_intData.find(_name);
        if(res == m_intData.end())
        {
            LOG_INFO() << "Failed to find data on key: " << _name;
            return 0;
        }
        else
            return res->second;
    }

    uint32_t GetUInt(const std::string& _name)
    {
        auto res = m_uintData.find(_name);
        if(res == m_uintData.end())
        {
            LOG_INFO() << "Failed to find data on key: " << _name;
            return 0;
        }
        else
            return res->second;
    }

    float GetFloat(const std::string& _name)
    {
        auto res = m_floatData.find(_name);
        if(res == m_floatData.end())
        {
            LOG_INFO() << "Failed to find data on key: " << _name;
            return 0.f;
        }
        else
            return res->second;
    }

	// SET Functions
    void Set(const std::string& _name, Layer<T, Mat>* _data)
    {
        if(m_LayerData.find(_name) != m_LayerData.end())
        {
            LOG_INFO() << "Overwriting data stored in key: " << _name;
        }
        m_LayerData.insert_or_assign(_name, _data);
    }

    void Set(const std::string& _name, Datablob<T, Mat>* _data)
    {
        if(m_ChildBlobData.find(_name) != m_ChildBlobData.end())
        {
            LOG_INFO() << "Overwriting data stored in key: " << _name;
        }
        m_ChildBlobData.insert_or_assign(_name, _data);
    }

    void Set(const std::string& _name, const MatrixRef& _data)
    {
        auto existing = m_matrixRefData.find(_name);
        bool sameHandle = false;
        if (existing != m_matrixRefData.end())
        {
            sameHandle = (_data.getHandle() == existing->second.getHandle());
            if(!sameHandle)
                LOG_INFO() << "Overwriting data stored in key: " << _name;
        }

        if(!sameHandle)
        {
            m_matrixRefData.insert_or_assign(_name, _data);
        }
    }

    void Set(const std::string& _name, MatrixHandle _handle)
    {
        if (_handle == InvalidMatrixHandle)
        {
            LOG_INFO() << "SetHandle: invalid handle for key: " << _name;
            return;
        }
        auto existing = m_matrixRefData.find(_name);
        bool sameHandle = false;
        if (    existing != m_matrixRefData.end() )
        {
            sameHandle = (_handle == existing->second.getHandle());
            if(!sameHandle)
                LOG_INFO() << "Overwriting data stored in key: " << _name;
        }
        if(!sameHandle)
        {
            MatrixRef ref = MatrixManager<T, Mat>::Instance().Acquire(_handle);
            if (!ref)
            {
                LOG_INFO() << "SetHandle: failed to acquire handle for key: " << _name;
                return;
            }
            m_matrixRefData.insert_or_assign(_name, std::move(ref));
        }
    }

    void Set(const std::string& _name, Mat* _data)
    {
        MatrixManager<T, Mat>& inst = MatrixManager<T, Mat>::Instance();

        auto handle = inst.GetHandle(_data);
        if(handle != InvalidMatrixHandle)
        {
            Set(_name, handle);
        }
        else
        {
            assert(false);
        }

    //    if (_data == nullptr)
    //    {
    //        LOG_INFO() << "Set: null matrix for key: " << _name;
    //        return;
    //    }
    //
    //    Mat* mat = dynamic_cast<Mat*>(_data);
    //    if (mat == nullptr)
    //    {
    //        LOG_INFO() << "Set: matrix type mismatch for key: " << _name;
    //        return;
    //    }
    //
    //    MatrixHandle handle = MatrixManager<T, Mat>::Instance().GetHandle(mat);
    //    if (handle == InvalidMatrixHandle)
    //    {
    //        LOG_INFO() << "Set: matrix not managed for key: " << _name;
    //        return;
    //    }
    //
    //    SetHandle(_name, handle);
    }

    void Set(const std::string& _name, int _data)
    {
        if(m_intData.find(_name) != m_intData.end())
        {
            LOG_INFO() << "Overwriting data stored in key: " << _name;
        }
        m_intData[_name] = _data;
    }

    void Set(const std::string& _name, uint32_t _data)
    {
        if(m_uintData.find(_name) != m_uintData.end())
        {
            LOG_INFO() << "Overwriting data stored in key: " << _name;
        }
        m_uintData[_name] = _data;
    }

    void Set(const std::string& _name, float _data)
    {
        if(m_floatData.find(_name) != m_floatData.end())
        {
            LOG_INFO() << "Overwriting data stored in key: " << _name;
        }
        m_floatData[_name] = _data;
    }
};
