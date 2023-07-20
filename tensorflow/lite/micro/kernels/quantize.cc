/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "quantize.h"

#include "../../core/c/common.h"
#include "../../kernels/internal/quantization_util.h"
#include "../../kernels/internal/tensor_ctypes.h"
#include "../../kernels/kernel_util.h"
#include "kernel_util.h"
#include "../micro_utils.h"

namespace tflite {
namespace {
/*
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  (void)buffer;
  (void)length;
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataQuantizeReference));
}

*/
}  // namespace

/*
TFLMRegistration Register_QUANTIZE() {
  return tflite::micro::RegisterOp(Init, PrepareQuantizeReference,
                                   EvalQuantizeReference);
}
*/
}  // namespace tflite
