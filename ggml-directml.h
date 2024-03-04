#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_DML_NAME "DirectML"
#define GGML_DML_MAX_DEVICES 1

struct ggml_directml_device {
    int index;
};

GGML_API void ggml_init_directml(void);

GGML_API ggml_backend_t ggml_backend_directml_init(int device);

GGML_API ggml_backend_buffer_type_t ggml_backend_directml_buffer_type(int device);

#ifdef  __cplusplus
}
#endif
