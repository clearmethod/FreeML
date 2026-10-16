#include <metal_stdlib>
using namespace metal;
struct MatrixParams {
    uint dimx, dimy, dimz, offset;
    uint uniqueId, channels, batch, reserved;
    uint optionalParams[16];
};
inline uint pixel(constant MatrixParams& p,uint b,uint c,uint x,uint y)
{ return p.offset+((b*p.channels+c)*p.dimy+y)*p.dimx+x; }
inline int coordinate(uint output,uint k,constant MatrixParams& p,bool transposed)
{
    const int s=int(p.optionalParams[1]),d=int(p.optionalParams[2]),pad=int(p.optionalParams[3]);
    const int position=transposed ? int(output)+pad-int(k)*d : int(output)*s+int(k)*d-pad;
    if(transposed && (position<0 || position%s)) return -1;
    return transposed ? position/s : position;
}
kernel void convolution_forward(device float* output [[buffer(0)]],device const float* input [[buffer(1)]],
    device const float* weights [[buffer(2)]],device const float* bias [[buffer(3)]],
    constant MatrixParams& o [[buffer(4)]],constant MatrixParams& i [[buffer(5)]],
    constant MatrixParams& k [[buffer(6)]],constant MatrixParams& a [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    if(index>=o.batch*o.dimx*o.dimy) return;
    const uint x=index%o.dimx,y=(index/o.dimx)%o.dimy,b=index/(o.dimx*o.dimy);
    const uint oc=o.optionalParams[0];const bool transposed=o.optionalParams[4]!=0u;
    float sum=o.optionalParams[5] ? bias[a.offset+oc] : 0.0f;
    for(uint c=0;c<i.channels;++c) for(uint ky=0;ky<k.dimy;++ky) for(uint kx=0;kx<k.dimx;++kx) {
        const int ix=coordinate(x,kx,o,transposed),iy=coordinate(y,ky,o,transposed);
        if(ix>=0 && iy>=0 && ix<int(i.dimx) && iy<int(i.dimy))
            sum+=input[pixel(i,b,c,uint(ix),uint(iy))]*weights[pixel(k,0u,c,kx,ky)];
    }
    output[pixel(o,b,oc,x,y)]=sum;
}
kernel void convolution_weights_backwards(device float* output [[buffer(0)]],device const float* input [[buffer(1)]],
    device const float* weights [[buffer(2)]],device const float* error [[buffer(3)]],
    constant MatrixParams& o [[buffer(4)]],constant MatrixParams& i [[buffer(5)]],
    constant MatrixParams& k [[buffer(6)]],constant MatrixParams& e [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    if(index>=k.channels*k.dimx*k.dimy) return;
    const uint kx=index%k.dimx,ky=(index/k.dimx)%k.dimy,c=index/(k.dimx*k.dimy);
    const uint oc=o.optionalParams[0];const bool transposed=o.optionalParams[4]!=0u;
    float sum=0;
    for(uint b=0;b<i.batch;++b) for(uint y=0;y<e.dimy;++y) for(uint x=0;x<e.dimx;++x) {
        const int ix=coordinate(x,kx,o,transposed),iy=coordinate(y,ky,o,transposed);
        if(ix>=0 && iy>=0 && ix<int(i.dimx) && iy<int(i.dimy))
            sum+=input[pixel(i,b,c,uint(ix),uint(iy))]*error[pixel(e,b,oc,x,y)];
    }
    output[o.offset+index]+=sum;
}
kernel void convolution_input_backwards(device float* output [[buffer(0)]],device const float* input [[buffer(1)]],
    device const float* weights [[buffer(2)]],device const float* error [[buffer(3)]],
    constant MatrixParams& o [[buffer(4)]],constant MatrixParams& i [[buffer(5)]],
    constant MatrixParams& k [[buffer(6)]],constant MatrixParams& e [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    if(index>=i.batch*i.channels*i.dimx*i.dimy) return;
    const uint x=index%i.dimx,y=(index/i.dimx)%i.dimy;
    const uint c=(index/(i.dimx*i.dimy))%i.channels,b=index/(i.dimx*i.dimy*i.channels);
    const uint oc=o.optionalParams[0];const bool transposed=o.optionalParams[4]!=0u;
    float sum=0;
    for(uint ky=0;ky<k.dimy;++ky) for(uint kx=0;kx<k.dimx;++kx) {
        const int ox=coordinate(x,kx,o,!transposed),oy=coordinate(y,ky,o,!transposed);
        if(ox>=0 && oy>=0 && ox<int(e.dimx) && oy<int(e.dimy))
            sum+=weights[pixel(k,0u,c,kx,ky)]*error[pixel(e,b,oc,uint(ox),uint(oy))];
    }
    output[o.offset+index]+=sum;
}
