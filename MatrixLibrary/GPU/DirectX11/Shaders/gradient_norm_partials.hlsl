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
void CSMain(uint3 group : SV_GroupID,uint tid : SV_GroupIndex) {
    uint count=in_dimx*in_dimy*in_dimz;
    uint groupIndex=group.x+group.y*65535u, start=groupIndex*1024u;
    if(start>=count) return;
    float2 sum=float2(0.0f,0.0f);
    for(uint j=0u;j<4u;++j) {
        uint i=start+tid+j*256u;
        if(i<count) sum=combine(sum,float2(abs(In[in_offset+i]),1.0f));
    }
    reduction[tid]=sum; GroupMemoryBarrierWithGroupSync();
    for(uint stride=128u;stride>0u;stride>>=1u) {
        if(tid<stride) reduction[tid]=combine(reduction[tid],reduction[tid+stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if(tid==0u) {
        uint index=out_offset+2u*(out_optionalParams[0].x+groupIndex);
        Out[index]=reduction[0].x; Out[index+1u]=reduction[0].y;
    }
}
