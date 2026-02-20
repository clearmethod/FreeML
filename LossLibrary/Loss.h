#pragma once

template<class T, class MatT>
class LossBase
{
    public:
    virtual ~LossBase() = default;
    virtual const char* GetName() const                                 = 0;
    virtual T       Loss    (MatT* target, MatT* prediction)            = 0;
    virtual void    Gradient(MatT* out, MatT* target, MatT* prediction) = 0;
};
