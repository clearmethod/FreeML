#include <MatrixLibrary/GPU/Metal/MatrixMetal.h>
#include <MatrixLibrary/GPU/Metal/MatrixMetal_Functions.h>
#include <MatrixLibrary/MatrixBase_Functions.h>
#include <MatrixLibrary/MatrixManager.h>

#include <ToolsLibrary/Logger.h>

#ifdef __APPLE__

int main()
{
    using T = float;
    using MatType = MatrixMetal<T>;
    using MatrixRef = typename MatrixManager<T, MatType>::MatrixRef;

    MetalManager* metal = MetalManager::Instance();
    if (!metal || !metal->GetQueue())
    {
        LOG_INFO() << "No Metal device available.";
        return 1;
    }

    auto& matrices = MatrixManager<T, MatType>::Instance();

    MatrixRef a = matrices.AllocateMatrix(MatrixShape::Matrix(3,2), "Metal_A");
    MatrixRef b = matrices.AllocateMatrix(MatrixShape::Matrix(2,3), "Metal_B");
    MatrixRef c = matrices.AllocateMatrix(MatrixShape::Matrix(2,2), "Metal_A_mul_B");

    for (uint32_t y = 0; y < a->GetHeight(); ++y)
    {
        for (uint32_t x = 0; x < a->GetWidth(); ++x)
        {
            a->SetValue(x, y, static_cast<T>((y * a->GetWidth()) + x + 1u));
        }
    }

    for (uint32_t y = 0; y < b->GetHeight(); ++y)
    {
        for (uint32_t x = 0; x < b->GetWidth(); ++x)
        {
            b->SetValue(x, y, static_cast<T>(10u + (y * b->GetWidth()) + x));
        }
    }

    a->Unmap();
    b->Unmap();

    LOG_INFO() << "First Metal matrix:\n" << a->GetString();
    LOG_INFO() << "Second Metal matrix:\n" << b->GetString();

    Mul(c.get(), a.get(), b.get());
    LOG_INFO() << "Metal matrix multiply result:\n" << c->GetString();
    LOG_INFO() << "Expected result:\n[ 76.000,82.000]\n[ 184.000,199.000]\n";

    LOG_INFO() << matrices.GetString(true);
    LOG_INFO() << MetalManager::Instance()->GetString();

    return 0;
}

#else

int main()
{
    return 0;
}

#endif
