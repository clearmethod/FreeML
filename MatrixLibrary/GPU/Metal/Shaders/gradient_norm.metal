#include <metal_stdlib>
using namespace metal;
struct MatrixParams {
    uint dimx, dimy, dimz, offset;
    uint uniqueId, pad0, pad1, pad2;
    uint optionalParams[16];
};
float2 combine(float2 a, float2 b) {
    float maximum = max(a.x,b.x);
    if (maximum == 0.0f) return float2(0.0f);
    float ar = a.x/maximum, br = b.x/maximum;
    return float2(maximum,a.y*ar*ar+b.y*br*br);
}
kernel void gradient_norm_partials(device float* partials [[buffer(0)]],
    device const float* gradient [[buffer(1)]],
    constant MatrixParams& p [[buffer(2)]], constant MatrixParams& g [[buffer(3)]],
    constant uint& destination [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 shared[256];
    uint count = g.dimx*g.dimy*g.dimz, start = group*1024u;
    float2 sum(0.0f);
    for (uint j=0u;j<4u;++j) {
        uint i=start+tid+j*256u;
        if(i<count) sum=combine(sum,float2(fabs(gradient[g.offset+i]),1.0f));
    }
    shared[tid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128u;stride>0u;stride>>=1u) {
        if(tid<stride) shared[tid]=combine(shared[tid],shared[tid+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0u) {
        uint index=p.offset+2u*(destination+group);
        partials[index]=shared[0].x; partials[index+1u]=shared[0].y;
    }
}
kernel void gradient_norm_scale(device float* scale [[buffer(0)]],
    device const float* partials [[buffer(1)]],
    constant MatrixParams& s [[buffer(2)]], constant MatrixParams& p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 shared[256];
    float2 sum(0.0f);
    for(uint i=tid;i<s.optionalParams[1];i+=256u)
        sum=combine(sum,float2(partials[p.offset+2u*i],partials[p.offset+2u*i+1u]));
    shared[tid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128u;stride>0u;stride>>=1u) {
        if(tid<stride) shared[tid]=combine(shared[tid],shared[tid+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0u) {
        float norm=shared[0].x*sqrt(shared[0].y);
        scale[s.offset]=min(1.0f,as_type<float>(s.optionalParams[0])/(norm+1e-6f));
    }
}
