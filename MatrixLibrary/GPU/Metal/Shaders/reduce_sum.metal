#include <metal_stdlib>
using namespace metal;
struct MatrixParams {
    uint dimx, dimy, dimz, offset;
    uint uniqueId, channels, batch, reserved;
    uint optionalParams[16];
};
kernel void reduce_sum(device float* output [[buffer(0)]],
                       device const float* input [[buffer(1)]],
                       constant MatrixParams& o [[buffer(2)]],
                       constant MatrixParams& i [[buffer(3)]],
                       uint group [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]])
{
    uint coord=group;
    const uint x=coord%o.dimx;coord/=o.dimx;
    const uint y=coord%o.dimy;coord/=o.dimy;
    const uint c=coord%o.channels, b=coord/o.channels;
    const uint axes=o.optionalParams[0];
    const uint rw=(axes&4u)?i.dimx:1u, rh=(axes&8u)?i.dimy:1u;
    const uint rc=(axes&2u)?i.channels:1u, rb=(axes&1u)?i.batch:1u;
    float sum=0;
    for(uint r=lane;r<rw*rh*rc*rb;r+=256u) {
        uint position=r;
        const uint ix=(axes&4u)?position%rw:x;position/=rw;
        const uint iy=(axes&8u)?position%rh:y;position/=rh;
        const uint ic=(axes&2u)?position%rc:c;position/=rc;
        const uint ib=(axes&1u)?position:b;
        sum+=input[i.offset+((ib*i.channels+ic)*i.dimy+iy)*i.dimx+ix];
    }
    threadgroup float shared[256];shared[lane]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128u;stride;stride/=2u) {
        if(lane<stride) shared[lane]+=shared[lane+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!lane) output[o.offset+group]=shared[0];
}
