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
#include "../micro_context.h"
#include "../micro_graph.h"
#include "../../schema/schema_generated.h"

namespace tflite {

namespace {

struct OpData {
  int init_subgraph_index;
  bool has_run;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  (void)buffer;
  (void)length;
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const auto* params =
      reinterpret_cast<const TfLiteCallOnceParams*>(node->builtin_data);
  op_data->init_subgraph_index = params->init_subgraph_index;
  op_data->has_run = false;

  TF_LITE_ENSURE(context, NumInputs(node) == 0);
  TF_LITE_ENSURE(context, NumOutputs(node) == 0);

  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
  MicroGraph& graph_info = micro_context->graph();

  TF_LITE_ENSURE(context,
                 op_data->init_subgraph_index < graph_info.NumSubgraphs());

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  // Call once only runs one time then is a no-op for every subsequent call.
  if (op_data->has_run) {
    return kTfLiteOk;
  }

  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
  MicroGraph& graph_info = micro_context->graph();

  TF_LITE_ENSURE_OK(context,
                    graph_info.InvokeSubgraph(op_data->init_subgraph_index));

  op_data->has_run = true;

  return kTfLiteOk;
}

}  // namespace.

TFLMRegistration Register_CALL_ONCE() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
