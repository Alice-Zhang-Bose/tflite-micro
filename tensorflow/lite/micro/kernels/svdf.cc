/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "svdf.h"

#include <math.h>

#include "../../core/c/builtin_op_data.h"
#include "../../core/c/common.h"
#include "../../kernels/internal/common.h"
#include "../../kernels/internal/quantization_util.h"
#include "../../kernels/internal/tensor_ctypes.h"
#include "../../kernels/kernel_util.h"
#include "../../kernels/op_macros.h"
#include "activation_utils.h"
#include "kernel_util.h"
#include "../micro_log.h"
#include "../micro_utils.h"

namespace tflite {
namespace {
/*

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  (void)buffer;
  (void)length;
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataSvdf));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataSvdf& data = *(static_cast<const OpDataSvdf*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsTimeTensor);
  // TODO(#1751): account for optional bias tensor
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 5)
          ? tflite::micro::GetEvalInput(context, node, kSvdfBiasTensor)
          : nullptr;
  TfLiteEvalTensor* activation_state = tflite::micro::GetMutableEvalInput(
      context, node, kSvdfInputActivationStateTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSvdfOutputTensor);

  switch (weights_feature->type) {
    case kTfLiteFloat32: {
      EvalFloatSvdfReference(
          context, node, input, weights_feature, weights_time, bias, params,
          data.scratch_tensor_index, activation_state, output);
      break;
    }

    case kTfLiteInt8: {
      switch (weights_time->type) {
        case kTfLiteInt16: {
          EvalInt16SvdfReference(context, node, input, weights_feature,
                                 weights_time, bias, params, activation_state,
                                 output, data);
          break;
        }
        case kTfLiteInt8: {
          EvalInt8SvdfReference(context, node, input, weights_feature,
                                weights_time, bias, params, activation_state,
                                output, data);
          break;
        }
        default:
          MicroPrintf("Type %s not currently supported.",
                      TfLiteTypeGetName(weights_time->type));
          return kTfLiteError;
      }
      break;
    }

    default:
      MicroPrintf("Type %s not currently supported.",
                  TfLiteTypeGetName(weights_feature->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_SVDF() {
  return tflite::micro::RegisterOp(Init, PrepareSvdf, Eval);
}
*/
}
}  // namespace tflite
