cbuffer OutParams : register(b0) {
    uint out_width, out_height, out_planes, out_offset;
    uint out_id, out_channels, out_batch, out_reserved;
    uint4 out_optional[4];
};
cbuffer InputParams : register(b1) {
    uint input_width, input_height, input_planes, input_offset;
    uint input_id, input_channels, input_batch, input_reserved;
};
RWStructuredBuffer<float> output : register(u0);
StructuredBuffer<float> input : register(t0);
groupshared float shared[256];
[numthreads(256,1,1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index=group.x+group.y*out_optional[0].y;
    if(index>=out_width*out_height*out_planes) return;
    uint coord=index;
    uint x=coord%out_width;coord/=out_width;
    uint y=coord%out_height;coord/=out_height;
    uint c=coord%out_channels, b=coord/out_channels;
    uint axes=out_optional[0].x;
    uint rw=(axes&4u)?input_width:1u, rh=(axes&8u)?input_height:1u;
    uint rc=(axes&2u)?input_channels:1u, rb=(axes&1u)?input_batch:1u;
    float sum=0;
    for(uint r=lane;r<rw*rh*rc*rb;r+=256u) {
        uint position=r;
        uint ix=(axes&4u)?position%rw:x;position/=rw;
        uint iy=(axes&8u)?position%rh:y;position/=rh;
        uint ic=(axes&2u)?position%rc:c;position/=rc;
        uint ib=(axes&1u)?position:b;
        sum+=input[input_offset+((ib*input_channels+ic)*input_height+iy)*input_width+ix];
    }
    shared[lane]=sum;GroupMemoryBarrierWithGroupSync();
    for(uint stride=128u;stride;stride/=2u) {
        if(lane<stride) shared[lane]+=shared[lane+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if(!lane) output[out_offset+index]=shared[0];
}
