
#define ROOT_SIG_DEF "DescriptorTable(UAV(u0, numDescriptors=3, flags=DATA_VOLATILE | DESCRIPTORS_VOLATILE)), RootConstants(num32BitConstants=2, b0)"

#define K_QUANTS_PER_ITERATION 2
#define MAX_COLS_PER_THREAD_GROUP (2 / K_QUANTS_PER_ITERATION)

#if 16 % K_QUANTS_PER_ITERATION != 0
#error "16 must be divisible by K_QUANTS_PER_ITERATION"
#endif

#define WARP_SIZE 32
#define QK_K 256

#define UnpackXbitsFromYbits(packed32Bit, index, X, Y) (((packed32Bit) << ((Y-X) - (index)*X)) >> (Y-X))
#define LoadXbitsFromYbits(buf, index, X, Y) UnpackXbitsFromYbits(buf[index/(Y/X)], index%(Y/X), X, Y)
#define Load8Bits(buf, index) LoadXbitsFromYbits(buf, index, 8, 32)

struct PackedData {
    uint32_t ql[QK_K/8];     // quants, lower 4 bits
    uint32_t qh[QK_K/16];    // quants, upper 2 bits
    int32_t scales[QK_K/64]; // scales, quantized with 8 bits
    float16_t d;             // super-block scale
};

RWStructuredBuffer<TIN> matA : register(u0);
RWStructuredBuffer<PackedData> matB : register(u1);
RWStructuredBuffer<TOUT> output : register(u2);

cbuffer Constants
{
    uint nrows;
    uint ncols;
};

[RootSignature(ROOT_SIG_DEF)]
[numthreads(WARP_SIZE, 1, MAX_COLS_PER_THREAD_GROUP)]
[WaveSize(WARP_SIZE)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadId)
{
    const int row = groupId.x * MAX_COLS_PER_THREAD_GROUP + threadId.z;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row * num_blocks_per_row;

#if QK_K == 256

    const int tid = threadId.x / K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadId.x % K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const int step = 16 / K_QUANTS_PER_ITERATION;          // 16 or 8

    const int im = tid / step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int inn = tid - step*im;                        // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION * inn;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * inn;                               // 0, 4, 8, ..., 28
    const int is = inn / 4;
#endif

    const int ql_offset = 64 * im + l0;
    const int qh_offset = 32 * im + l0;
    const int s_offset = 8 * im + is;
    const int y_offset = 128 * im + l0;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        PackedData packed_value = matB[ib0 + i];
        const uint mat_a_offset = i * QK_K + y_offset;
        const float d = packed_value.d;

#if K_QUANTS_PER_ITERATION == 1
        float sum = matA[mat_a_offset] * (float16_t)Load8Bits(packed_value.scales, s_offset) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset) & 0xF) | ((Load8Bits(packed_value.qh, qh_offset) & 0x03) << 4)) - 32)
                  + matA[mat_a_offset + 16] * (float16_t)Load8Bits(packed_value.scales, s_offset + 1) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset + 16) & 0xF) | ((Load8Bits(packed_value.qh, qh_offset + 16) & 0x03) << 4)) - 32)
                  + matA[mat_a_offset + 32] * (float16_t)Load8Bits(packed_value.scales, s_offset + 2) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset + 32) & 0xF) | ((Load8Bits(packed_value.qh, qh_offset) & 0x0c) << 2)) - 32)
                  + matA[mat_a_offset + 48] * (float16_t)Load8Bits(packed_value.scales, s_offset + 3) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset + 48) & 0xF) | ((Load8Bits(packed_value.qh, qh_offset + 16) & 0x0c) << 2)) - 32)
                  + matA[mat_a_offset + 64] * (float16_t)Load8Bits(packed_value.scales, s_offset + 4) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset)  >> 4) | ((Load8Bits(packed_value.qh, qh_offset) & 0x30) >> 0)) - 32)
                  + matA[mat_a_offset + 80] * (float16_t)Load8Bits(packed_value.scales, s_offset + 5) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset + 16)  >> 4) | ((Load8Bits(packed_value.qh, qh_offset + 16) & 0x30) >> 0)) - 32)
                  + matA[mat_a_offset + 96] * (float16_t)Load8Bits(packed_value.scales, s_offset + 6) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset + 32)  >> 4) | ((Load8Bits(packed_value.qh, qh_offset) & 0xc0) >> 2)) - 32)
                  + matA[mat_a_offset + 112] * (float16_t)Load8Bits(packed_value.scales, s_offset + 7) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset + 48)  >> 4) | ((Load8Bits(packed_value.qh, qh_offset + 16) & 0xc0) >> 2)) - 32);
        tmp += sum;
#else
        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            sum += matA[mat_a_offset + l] * (float16_t)Load8Bits(packed_value.scales, s_offset) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset + l) & 0xF) | (((Load8Bits(packed_value.qh, qh_offset + l) >> 0) & 3) << 4)) - 32)
                 + matA[mat_a_offset + l + 32] * (float16_t)Load8Bits(packed_value.scales, s_offset + 2) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset + l + 32) & 0xF) | (((Load8Bits(packed_value.qh, qh_offset + l) >> 2) & 3) << 4)) - 32)
                 + matA[mat_a_offset + l + 64] * (float16_t)Load8Bits(packed_value.scales, s_offset + 4) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset + l)  >> 4) | (((Load8Bits(packed_value.qh, qh_offset + l) >> 4) & 3) << 4)) - 32)
                 + matA[mat_a_offset + l + 96] * (float16_t)Load8Bits(packed_value.scales, s_offset + 6) * d * ((int32_t)((Load8Bits(packed_value.ql, ql_offset + l + 32)  >> 4) | (((Load8Bits(packed_value.qh, qh_offset + l) >> 6) & 3) << 4)) - 32);
        }
        tmp += sum;
#endif

    }

#else

    const int tid = threadId.x / (2 * K_QUANTS_PER_ITERATION);  // 0...7
    const int ix = threadId.x % (2 * K_QUANTS_PER_ITERATION);  // 0...3
    const int step = tid * K_QUANTS_PER_ITERATION;
    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2 * K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = x[i].ql + step;
        const uint8_t * qh = x[i].qh + step;
        const int8_t  * s  = x[i].scales;

        const float d = x[i+0].d;

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
    tmp = WaveActiveSum(tmp);

    if (tid == 0) {
        output[row] = (TOUT)tmp;
    }
}
