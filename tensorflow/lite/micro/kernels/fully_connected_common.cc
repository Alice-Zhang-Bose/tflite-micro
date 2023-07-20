/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "fully_connected.h"

#include "../../core/c/builtin_op_data.h"
#include "../../core/c/common.h"
#include "../../kernels/internal/common.h"
#include "../../kernels/internal/quantization_util.h"
#include "../../kernels/internal/reference/fully_connected.h"
#include "../../kernels/internal/reference/integer_ops/fully_connected.h"
#include "../../kernels/internal/tensor_ctypes.h"
#include "../../kernels/kernel_util.h"
#include "kernel_util.h"

namespace tflite {

const int kFullyConnectedInputTensor = 0;
const int kFullyConnectedWeightsTensor = 1;
const int kFullyConnectedBiasTensor = 2;
const int kFullyConnectedOutputTensor = 0;

FullyConnectedParams FullyConnectedParamsQuantized(
    const OpDataFullyConnected& op_data) {
  FullyConnectedParams op_params;
  op_params.input_offset = -op_data.input_zero_point;
  op_params.weights_offset = -op_data.filter_zero_point;
  op_params.output_offset = op_data.output_zero_point;
  op_params.output_multiplier = op_data.output_multiplier;
  op_params.output_shift = op_data.output_shift;
  op_params.quantized_activation_min = op_data.output_activation_min;
  op_params.quantized_activation_max = op_data.output_activation_max;
  return op_params;
}

FullyConnectedParams FullyConnectedParamsFloat(
    TfLiteFusedActivation activation) {
  FullyConnectedParams op_params;
  CalculateActivationRange(activation, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  return op_params;
}

TfLiteStatus CalculateOpDataFullyConnected(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    OpDataFullyConnected* data) {
  if (data_type != kTfLiteFloat32) {
    //double real_multiplier = 0.0;
    // !! COMMENTING OUT BLOCK OF CODE BELOW DUE TO ERRORS!!
    // C:/git\walnut\3rdParty\tflite-micro\tensorflow\lite\micro\kernels/fully_connected_common.cc:62: undefined reference to `tflite::GetQuantizedConvolutionMultipler(TfLiteContext*, TfLiteTensor const*, TfLiteTensor const*, TfLiteTensor const*, TfLiteTensor*, double*)'
    //C:/usr/xtensa/XtDevTools/install/tools/RI-2021.8-win32/XtensaTools/bin/xtensa-elf-ld.exe: C:/git\walnut\3rdParty\tflite-micro\tensorflow\lite\micro\kernels/fully_connected_common.cc:77: undefined reference to `tflite::CalculateActivationRangeQuantized(TfLiteContext*, TfLiteFusedActivation, TfLiteTensor*, int*, int*)'

    /*
    
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
  */
    // Filter weights will always be symmetric quantized since we only support
    // int8 quantization. See
    // https://github.com/tensorflow/tensorflow/issues/44912 for additional
    // context.
    TFLITE_DCHECK(filter->params.zero_point == 0);

    data->input_zero_point = input->params.zero_point;
    data->filter_zero_point = filter->params.zero_point;
    data->output_zero_point = output->params.zero_point;

    // !! COMMENTING OUT DUE TO ERROR !!
    //C:/git\walnut\3rdParty\tflite-micro\tensorflow\lite\micro\kernels/fully_connected_common.cc:79: undefined reference to `tflite::CalculateActivationRangeQuantized(TfLiteContext*, TfLiteFusedActivation, TfLiteTensor*, int*, int*)'
  /*
    return CalculateActivationRangeQuantized(context, activation, output,
                                             &data->output_activation_min,
                                             &data->output_activation_max);
  */
  }
  return kTfLiteOk;
}

}  // namespace tflite
