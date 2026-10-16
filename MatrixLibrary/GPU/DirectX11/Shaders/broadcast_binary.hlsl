cbuffer OutParams : register(b0) {
    uint out_width, out_height, out_planes, out_offset;
    uint out_id, out_channels, out_batch, out_reserved;
    uint4 out_optional[4];
};
cbuffer LeftParams : register(b1) {
    uint left_width, left_height, left_planes, left_offset;
    uint left_id, left_channels, left_batch, left_reserved;
};
cbuffer RightParams : register(b2) {
    uint right_width, right_height, right_planes, right_offset;
    uint right_id, right_channels, right_batch, right_reserved;
};
RWStructuredBuffer<float> output : register(u0);
StructuredBuffer<float> left : register(t0);
StructuredBuffer<float> right : register(t1);
[numthreads(256,1,1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index=(group.x+group.y*out_optional[0].y)*256u+lane;
    if(index>=out_width*out_height*out_planes) return;
    uint coord=index,x=coord%out_width;coord/=out_width;
    uint y=coord%out_height;coord/=out_height;
    uint c=coord%out_channels,b=coord/out_channels;
    uint li=left_offset+(((left_batch==1u?0u:b)*left_channels+(left_channels==1u?0u:c))*left_height
        +(left_height==1u?0u:y))*left_width+(left_width==1u?0u:x);
    uint ri=right_offset+(((right_batch==1u?0u:b)*right_channels+(right_channels==1u?0u:c))*right_height
        +(right_height==1u?0u:y))*right_width+(right_width==1u?0u:x);
    uint flags=out_optional[0].x,operation=flags&3u;
    float a,bvalue;
    if(flags&4u) a=output[out_offset+index];else a=left[li];
    if(flags&8u) bvalue=output[out_offset+index];else bvalue=right[ri];
    output[out_offset+index]=operation==0u?a+bvalue:operation==1u?a-bvalue:a*bvalue;
}
