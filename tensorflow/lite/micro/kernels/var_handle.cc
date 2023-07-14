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

#include <stddef.h>

#include <cstring>

#include "../../core/c/builtin_op_data.h"
#include "../../core/c/common.h"
#include "../../kernels/internal/compatibility.h"
#include "../../kernels/kernel_util.h"
#include "kernel_util.h"
#include "../memory_helpers.h"
#include "../micro_graph.h"
#include "../micro_log.h"
#include "../micro_resource_variable.h"
#include "../../schema/schema_generated.h"

namespace tflite {

namespace {

struct OpData {
  int32_t resource_id;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  (void)buffer;
  (void)length;
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const auto* params =
      reinterpret_cast<const TfLiteVarHandleParams*>(node->builtin_data);

  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
  MicroGraph& graph_info = micro_context->graph();

  MicroResourceVariables* resources = graph_info.GetResourceVariables();
  if (resources == nullptr) {
    MicroPrintf(
        "VAR_HANDLE requires resource variables. Please create "
        "ResourceVariables and pass it to the interpreter.");
    return kTfLiteError;
  }
  op_data->resource_id =
      resources->CreateIdIfNoneFound(params->container, params->shared_name);
  if (op_data->resource_id < 0) {
    return kTfLiteError;
  }

  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TFLITE_DCHECK(output != nullptr);

  // Assign saved resource_id so this output tensor will always return the
  // correct resource id.
  output->data.i32 = &op_data->resource_id;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TFLITE_DCHECK(output != nullptr);

  // Assign saved resource_id so this output tensor will always return the
  // correct resource id.
  output->data.i32 = &op_data->resource_id;
  return kTfLiteOk;
}

}  // namespace.

TFLMRegistration Register_VAR_HANDLE() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
