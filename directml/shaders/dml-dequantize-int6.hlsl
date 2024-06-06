
#define ROOT_SIG_DEF "DescriptorTable(UAV(u0, numDescriptors=2, flags=DATA_VOLATILE | DESCRIPTORS_VOLATILE)), RootConstants(num32BitConstants=2, b0)"

#define QK_K 256

#if QK_K == 256
#define NUM_THREADS 64
#else
#define NUM_THREADS 32
#endif

#define UnpackXbitsFromYbits(packed32Bit, index, X, Y) (((packed32Bit) << ((Y-X) - (index)*X)) >> (Y-X))
#define LoadXbitsFromYbits(buf, index, X, Y) UnpackXbitsFromYbits(buf[index/(Y/X)], index%(Y/X), X, Y)
#define Load8Bits(buf, index) LoadXbitsFromYbits(buf, index, 8, 32)

struct PackedData {
    uint32_t ql[QK_K/8];     // quants, lower 4 bits
    uint32_t qh[QK_K/16];    // quants, upper 2 bits
    int32_t scales[QK_K/64]; // scales, quantized with 8 bits
    float16_t d;             // super-block scale
};

RWStructuredBuffer<PackedData> input : register(u0);
RWStructuredBuffer<TOUT> output : register(u1);

cbuffer Constants
{
};

[RootSignature(ROOT_SIG_DEF)]
[numthreads(NUM_THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadId)
{
    const PackedData packedValue = input[groupId.x];
    const float d = packedValue.d;

#if QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    const int ip = threadId.x / 32;   // ip is 0 or 1
    const int il = threadId.x - 32 * ip; // 0...32
    const int is = 8 * ip + il / 16;

    
    const uint output_offset = groupId.x * QK_K + 128 * ip + il;
    const uint ql_offset = 64 * ip + il;
    const uint32_t qh = Load8Bits(packedValue.qh, 32 * ip + il);

    output[output_offset] = (TOUT)(d * Load8Bits(packedValue.scales, is) * ((int32_t)((Load8Bits(packedValue.ql, ql_offset) & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    output[output_offset + 32] = (TOUT)(d * Load8Bits(packedValue.scales, is + 2) * ((int32_t)((Load8Bits(packedValue.ql, ql_offset + 32) & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    output[output_offset + 64] = (TOUT)(d * Load8Bits(packedValue.scales, is + 4) * ((int32_t)((Load8Bits(packedValue.ql, ql_offset) >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    output[output_offset + 96] = (TOUT)(d * Load8Bits(packedValue.scales, is + 6) * ((int32_t)((Load8Bits(packedValue.ql, ql_offset + 32) >> 4) | (((qh >> 6) & 3) << 4)) - 32));
#else

    // assume 32 threads
    const int ip  = threadId.x / 16;         // 0 or 1
    const int il  = threadId.x - 16 * ip;    // 0...15

    const uint output_offset = groupId.x * QK_K + 16 * ip + il;
    const uint32_t ql = Load8Bits(packedValue.ql, 16 * ip + il);
    const uint32_t qh = Load8Bits(packedValue.qh, il) >> (2 * ip);

    output[output_offset] = (TOUT)(d * Load8Bits(packedValue.scales, ip) * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    output[output_offset + 32] = (TOUT)(d * Load8Bits(packedValue.scales, ip + 2) * ((int8_t)((ql >> 4) | (((qh >> 4) & 3) << 4)) - 32));
#endif
}
