/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "../../../core/c/builtin_op_data.h"
#include "../../../core/c/common.h"
#include "../../../kernels/internal/common.h"
#include "../../../kernels/internal/quantization_util.h"
#include "../../../kernels/internal/reference/integer_ops/conv.h"
#include "../../../kernels/internal/tensor_ctypes.h"
#include "../../../kernels/kernel_util.h"
#include "../../../kernels/padding.h"
#include "../conv.h"
#include "../kernel_util.h"

namespace tflite {
namespace {

/*
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}
*/
}  // namespace.

TfLiteStatus ConvReferenceEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  const auto& op_data = *(reinterpret_cast<OpDataConv*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;

  reference_integer_ops::ConvPerChannel(
      ConvParamsQuantized(params, op_data),
      op_data.per_channel_output_multiplier, op_data.per_channel_output_shift,
      tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

// TODO(b/189981943): This variant can be used for a smaller binary
// since the optimized conv implementation currently adds a lot to
// the binary size (~30KB to text section).
/*
TFLMRegistration Register_CONV_2D_INT8REF() {
  return tflite::micro::RegisterOp(Init, ConvPrepare, ConvReferenceEvalInt8);
}
*/

}  // namespace tflite
