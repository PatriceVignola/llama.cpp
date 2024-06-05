@REM TODO (pavignol): Remove the debug options for release shader

@REM Copy shaders
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-copy.hlsl -T cs_6_2 -DTIN=float16_t -DTOUT=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-copy-16-16.h
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-copy.hlsl -T cs_6_2 -DTIN=float16_t -DTOUT=float -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-copy-16-32.h
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-copy.hlsl -T cs_6_2 -DTIN=float -DTOUT=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-copy-32-16.h
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-copy.hlsl -T cs_6_2 -DTIN=float -DTOUT=float -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-copy-32-32.h

@REM Quantization preprocessing shaders
.\build\packages\Microsoft.Direct3D.DXC.1.8.2405.17\build\native\bin\x64\dxc.exe directml\shaders\dml-quant-tensor-preprocessor.hlsl -T cs_6_2 -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Qembed_debug -Zi -Fh directml\generated-shaders\dml-quant-tensor-preprocessor.h