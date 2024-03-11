
#define ROOT_SIG_DEF "DescriptorTable(UAV(u0, numDescriptors=2, flags=DATA_VOLATILE | DESCRIPTORS_VOLATILE)), RootConstants(num32BitConstants=18, b0)"
#define NUM_THREADS 256

RWStructuredBuffer<TIN> input : register(u0);
RWStructuredBuffer<TOUT> output : register(u1);

cbuffer Constants
{
    uint4 inputSizes;
    uint4 outputSizes;
    uint4 inputStrides;
    uint4 outputStrides;
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
        const uint32_t4 outputIndices = GetCoordinatesFromLogicalIndex(globalIndex, outputSizes);
        const uint inputElementOffset = GetOffsetFromCoordinates(inputIndices, inputStrides);
        const uint outputElementOffset = GetOffsetFromCoordinates(outputIndices, outputStrides);
        output[outputElementOffset] = input[inputElementOffset];
    }
}