
#define ROOT_SIG_DEF "DescriptorTable(UAV(u0, numDescriptors=3, flags=DATA_VOLATILE | DESCRIPTORS_VOLATILE)), RootConstants(num32BitConstants=10, b0)"
#define NUM_THREADS 256

struct QuantizedElement
{
    float16_t scale;
    int16_t packedData[16];
};

RWStructuredBuffer<QuantizedElement> input : register(u0);
RWStructuredBuffer<int16_t> outputQuantizedData : register(u1);
RWStructuredBuffer<float16_t> outputScale : register(u2);

cbuffer Constants
{
    uint4 inputSizes;
    uint4 inputStrides;
    uint elementCount;
    uint startIndex;
};

inline uint4 GetCoordinatesFromLogicalIndex(uint globalIndex, uint4 sizes)
{
    uint4 coordinates;
    coordinates[3] = globalIndex % sizes[3]; globalIndex /= sizes[3];
    coordinates[2] = globalIndex % sizes[2]; globalIndex /= sizes[2];
    coordinates[1] = globalIndex % sizes[1];
    coordinates[0] = globalIndex / sizes[1];
    return coordinates;
}

inline uint GetOffsetFromCoordinates(uint32_t4 coordinates, uint32_t4 strides)
{
    return dot(coordinates, strides);
}

[RootSignature(ROOT_SIG_DEF)]
[numthreads(NUM_THREADS, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint globalIndex = dtid.x + startIndex;

    if (globalIndex < elementCount)
    {
        const uint32_t4 inputIndices = GetCoordinatesFromLogicalIndex(globalIndex, inputSizes);
        const uint inputElementOffset = GetOffsetFromCoordinates(inputIndices, inputStrides);
        QuantizedElement inputData = input[inputElementOffset];

        [unroll]
        for (uint i = 0; i < 16; ++i)
        {
            outputQuantizedData[globalIndex * 16 + i] = inputData.packedData[i];
        }

        outputScale[globalIndex] = inputData.scale;
    }
}