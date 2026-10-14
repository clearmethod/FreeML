cbuffer OutParams : register(b0) {
    uint out_dimx, out_dimy, out_dimz, out_offset;
    uint out_uniqueId, out_pad0, out_pad1, out_pad2;
    uint4 out_optionalParams[4];
};
cbuffer InParams : register(b1) {
    uint in_dimx, in_dimy, in_dimz, in_offset;
    uint in_uniqueId, in_pad0, in_pad1, in_pad2;
    uint4 in_optionalParams[4];
};
RWStructuredBuffer<float> Out : register(u0);
StructuredBuffer<float> In : register(t0);
groupshared float2 reduction[256];
float2 combine(float2 a,float2 b) {
    float maximum=max(a.x,b.x);
    if(maximum==0.0f) return float2(0.0f,0.0f);
    float ar=a.x/maximum, br=b.x/maximum;
    return float2(maximum,a.y*ar*ar+b.y*br*br);
}
[numthreads(256,1,1)]
void CSMain(uint tid : SV_GroupIndex) {
    float2 sum=float2(0.0f,0.0f);
    for(uint i=tid;i<out_optionalParams[0].y;i+=256u)
        sum=combine(sum,float2(In[in_offset+2u*i],In[in_offset+2u*i+1u]));
    reduction[tid]=sum; GroupMemoryBarrierWithGroupSync();
    for(uint stride=128u;stride>0u;stride>>=1u) {
        if(tid<stride) reduction[tid]=combine(reduction[tid],reduction[tid+stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if(tid==0u) {
        float norm=reduction[0].x*sqrt(reduction[0].y);
        Out[out_offset]=min(1.0f,asfloat(out_optionalParams[0].x)/(norm+1e-6f));
    }
}
