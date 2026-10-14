#include <metal_stdlib>
using namespace metal;

struct MatrixParams
{
    uint dimx;
    uint dimy;
    uint dimz;
    uint offset;
    uint uniqueId;
    uint pad0;
    uint pad1;
    uint pad2;
};

kernel void sum_columns_parallel(device float* out [[buffer(0)]],
                        device const float* in [[buffer(1)]],
                        constant MatrixParams& outParams [[buffer(2)]],
                        constant MatrixParams& inParams [[buffer(3)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint3 lane [[thread_position_in_threadgroup]])
{
    uint col=group.x*32u+lane.x;
    threadgroup float partial[8][32];
    float sum=0.0f;
    if(col<outParams.dimx && col<inParams.dimx)
        for(uint row=lane.y;row<inParams.dimy;row+=8u)
            sum+=in[inParams.offset+row*inParams.dimx+col];
    partial[lane.y][lane.x]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=4u;stride>0u;stride>>=1u) {
        if(lane.y<stride) partial[lane.y][lane.x]+=partial[lane.y+stride][lane.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane.y==0u && col<outParams.dimx && col<inParams.dimx)
        out[outParams.offset+col]=partial[0][lane.x];
}
