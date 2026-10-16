#include <metal_stdlib>
using namespace metal;
struct MatrixParams {
    uint dimx, dimy, dimz, offset;
    uint uniqueId, channels, batch, reserved;
    uint optionalParams[16];
};
inline uint broadcastIndex(constant MatrixParams& p,uint b,uint c,uint x,uint y)
{
    return p.offset+(((p.batch==1u?0u:b)*p.channels+(p.channels==1u?0u:c))*p.dimy
        +(p.dimy==1u?0u:y))*p.dimx+(p.dimx==1u?0u:x);
}
kernel void broadcast_binary(device float* output [[buffer(0)]],
                             device const float* left [[buffer(1)]],
                             device const float* right [[buffer(2)]],
                             constant MatrixParams& o [[buffer(3)]],
                             constant MatrixParams& l [[buffer(4)]],
                             constant MatrixParams& r [[buffer(5)]],
                             uint index [[thread_position_in_grid]])
{
    if(index>=o.dimx*o.dimy*o.dimz) return;
    uint coord=index;const uint x=coord%o.dimx;coord/=o.dimx;
    const uint y=coord%o.dimy;coord/=o.dimy;
    const uint c=coord%o.channels,b=coord/o.channels;
    const uint flags=o.optionalParams[0],operation=flags&3u;
    const float a=(flags&4u)?output[o.offset+index]:left[broadcastIndex(l,b,c,x,y)];
    const float bvalue=(flags&8u)?output[o.offset+index]:right[broadcastIndex(r,b,c,x,y)];
    output[o.offset+index]=operation==0u?a+bvalue:operation==1u?a-bvalue:a*bvalue;
}
