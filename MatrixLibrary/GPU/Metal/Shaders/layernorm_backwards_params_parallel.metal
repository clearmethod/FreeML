#include <metal_stdlib>
using namespace metal;

struct MatrixParams
{
    uint dimx;
    uint dimy;
    uint dimz;
    uint offset;
    uint uniqueId;
    uint channels;
    uint batch;
    uint reserved;
};

kernel void layernorm_backwards_params_parallel(device float* dgamma [[buffer(0)]],
                                       device float* dbeta [[buffer(1)]],
                                       device const float* dy [[buffer(2)]],
                                       device const float* xhat [[buffer(3)]],
                                       constant MatrixParams& dgammaParams [[buffer(4)]],
                                       constant MatrixParams& dbetaParams [[buffer(5)]],
                                       constant MatrixParams& dyParams [[buffer(6)]],
                                       constant MatrixParams& xhatParams [[buffer(7)]],
                                       uint3 group [[threadgroup_position_in_grid]],
                        uint3 lane [[thread_position_in_threadgroup]])
{
    uint col=group.x*32u+lane.x;
    threadgroup float2 partial[8][32];
    float2 sum=float2(0.0f);
    const uint rows=dyParams.dimy*dyParams.dimz;
    if(col<dgammaParams.dimx)
        for(uint row=lane.y;row<rows;row+=8u) {
            uint z=row/dyParams.dimy, y=row-z*dyParams.dimy;
            uint dyBase=dyParams.offset+z*dyParams.dimx*dyParams.dimy+y*dyParams.dimx;
            uint xhatBase=xhatParams.offset+z*xhatParams.dimx*xhatParams.dimy+y*xhatParams.dimx;
            float value=dy[dyBase+col];
            sum+=float2(value*xhat[xhatBase+col],value);
        }
    partial[lane.y][lane.x]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=4u;stride>0u;stride>>=1u) {
        if(lane.y<stride) partial[lane.y][lane.x]+=partial[lane.y+stride][lane.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane.y==0u && col<dgammaParams.dimx) {
        dgamma[dgammaParams.offset+col]=partial[0][lane.x].x;
        dbeta[dbetaParams.offset+col]=partial[0][lane.x].y;
    }
}
