@REM TODO (pavignol): Remove the debug options for release shader

@REM Copy shaders
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-copy.hlsl -T cs_6_2 -DTIN=float16_t -DTOUT=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-copy-16-16.h
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-copy.hlsl -T cs_6_2 -DTIN=float16_t -DTOUT=float -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-copy-16-32.h
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-copy.hlsl -T cs_6_2 -DTIN=float -DTOUT=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-copy-32-16.h
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-copy.hlsl -T cs_6_0 -DTIN=float -DTOUT=float -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-copy-32-32.h

@REM TODO (pavignol): Have a version of the shaders not require CS 6.6 (i.e. not use WARP_SIZE)
@REM QuantizedGemmInt6 shaders
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-quantized-gemm-int6.hlsl -T cs_6_6 -DTIN=float16_t -DTOUT=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-quantized-gemm-int6-16-16.h
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-quantized-gemm-int6.hlsl -T cs_6_6 -DTIN=float16_t -DTOUT=float -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-quantized-gemm-int6-16-32.h
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-quantized-gemm-int6.hlsl -T cs_6_6 -DTIN=float -DTOUT=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-quantized-gemm-int6-32-16.h
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-quantized-gemm-int6.hlsl -T cs_6_6 -DTIN=float -DTOUT=float -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-quantized-gemm-int6-32-32.h

@REM DequantizeInt6 shaders
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-dequantize-int6.hlsl -T cs_6_2 -DTOUT=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-dequantize-int6-16.h
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-dequantize-int6.hlsl -T cs_6_2 -DTOUT=float -O3 -enable-16bit-types -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-dequantize-int6-32.h