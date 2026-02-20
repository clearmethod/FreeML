cbuffer MatOut_Params : register(b0)
{
    uint out_dimx;
    uint out_dimy;
    uint out_dimz;
    uint out_offset;
    uint out_uniqueId;
    uint out_pad0;
    uint out_pad1;
    uint out_pad2;
    uint4 out_optionalParams[4];
};

cbuffer MatLeft_Params : register(b1)
{
    uint left_dimx;
    uint left_dimy;
    uint left_dimz;
    uint left_offset;
    uint left_uniqueId;
    uint left_pad0;
    uint left_pad1;
    uint left_pad2;
    uint4 left_optionalParams[4];
};

cbuffer MatRight_Params : register(b2)
{
    uint right_dimx;
    uint right_dimy;
    uint right_dimz;
    uint right_offset;
    uint right_uniqueId;
    uint right_pad0;
    uint right_pad1;
    uint right_pad2;
    uint4 right_optionalParams[4];
};

RWStructuredBuffer<float> MatOut   : register(u0);
StructuredBuffer<float> MatLeft  : register(t1);
StructuredBuffer<float> MatRight : register(t2);

[numthreads(256, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint out_plane = out_dimx * out_dimy;
    uint out_count = out_plane * out_dimz;
    if (idx >= out_count)
        return;

    uint out_x = idx % out_dimx;
    uint out_y = (idx / out_dimx) % out_dimy;
    uint out_z = idx / out_plane;

    uint out_rowStride = out_optionalParams[0].x;
    uint out_colStride = out_optionalParams[0].y;
    uint left_rowStride = left_optionalParams[0].x;
    uint left_colStride = left_optionalParams[0].y;
    uint right_rowStride = right_optionalParams[0].x;
    uint right_colStride = right_optionalParams[0].y;

    if (out_rowStride == 0) out_rowStride = out_dimx;
    if (out_colStride == 0) out_colStride = 1u;
    if (left_rowStride == 0) left_rowStride = left_dimx;
    if (left_colStride == 0) left_colStride = 1u;
    if (right_rowStride == 0) right_rowStride = right_dimx;
    if (right_colStride == 0) right_colStride = 1u;

    uint out_planeStride = out_rowStride * out_dimy;
    uint left_planeStride = left_rowStride * left_dimy;
    uint right_planeStride = right_rowStride * right_dimy;

    uint out_base = out_offset + out_z * out_planeStride;
    uint left_base = left_offset + out_z * left_planeStride;
    uint right_base = right_offset + out_z * right_planeStride;

    float sum = 0.0f;
    for (uint k = 0; k < left_dimy; ++k)
    {
        uint left_idx = left_base + (k * left_rowStride) + out_y * left_colStride;
        uint right_idx = right_base + (k * right_rowStride) + out_x * right_colStride;
        sum += MatLeft[left_idx] * MatRight[right_idx];
    }

    MatOut[out_base + (out_y * out_rowStride) + out_x * out_colStride] = sum;
}
