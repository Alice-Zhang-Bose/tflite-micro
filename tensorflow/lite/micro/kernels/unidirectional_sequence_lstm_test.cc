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
#include "../../core/c/builtin_op_data.h"
#include "../../core/c/common.h"
#include "../../kernels/internal/portable_tensor_utils.h"
#include "../../kernels/internal/types.h"
#include "kernel_runner.h"
#include "lstm_shared.h"
#include "micro_ops.h"
#include "testdata/lstm_test_data.h"
#include "../test_helpers.h"
#include "../testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {
// Validate the output result array with golden values
template <typename T>
void ValidateResultGoldens(const T* golden, const T* output_data,
                           const int output_len, const float tolerance) {
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void TestUnidirectionalLSTMInteger(
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        node_contents) {
  const TFLMRegistration registration = Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  auto buildin_data = node_contents.BuiltinData();
  micro::KernelRunner runner(registration, node_contents.GetTensors(), 24 + 1,
                             node_contents.KernelInputs(),
                             node_contents.KernelOutputs(),
                             reinterpret_cast<void*>(&buildin_data));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const auto& quantization_settings = node_contents.QuantizationSettings();
  float dequantized_hidden_state[batch_size * state_dimension] = {};
  Dequantize(node_contents.GetHiddenStateData(), batch_size * state_dimension,
             quantization_settings.hidden_state.scale,
             quantization_settings.hidden_state.zero_point,
             dequantized_hidden_state);

  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        dequantized_hidden_state, batch_size * state_dimension,
                        hidden_state_tolerance);

  float dequantized_cell_state[batch_size * state_dimension] = {};
  Dequantize(node_contents.GetCellStateData(), batch_size * state_dimension,
             quantization_settings.cell_state.scale,
             quantization_settings.cell_state.zero_point,
             dequantized_cell_state);
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        dequantized_cell_state, batch_size * state_dimension,
                        cell_state_tolerance);

  float dequantized_output[batch_size * state_dimension * time_steps] = {};
  Dequantize(node_contents.GetOutputData(),
             batch_size * state_dimension * time_steps,
             quantization_settings.output.scale,
             quantization_settings.output.zero_point, dequantized_output);
  ValidateResultGoldens(eval_check_data.expected_output, dequantized_output,
                        batch_size * state_dimension, hidden_state_tolerance);
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestUnidirectionalLSTMFloat(
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& node_contents) {
  const TFLMRegistration registration = Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  auto buildin_data = node_contents.BuiltinData();
  micro::KernelRunner runner(registration, node_contents.GetTensors(), 24 + 1,
                             node_contents.KernelInputs(),
                             node_contents.KernelOutputs(),
                             reinterpret_cast<void*>(&buildin_data));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        node_contents.GetHiddenStateData(),
                        batch_size * state_dimension, hidden_state_tolerance);
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        node_contents.GetCellStateData(),
                        batch_size * state_dimension, cell_state_tolerance);
  ValidateResultGoldens(eval_check_data.expected_output,
                        node_contents.GetOutputData(),
                        batch_size * state_dimension, hidden_state_tolerance);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN
// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
#if !defined(XTENSA)
TF_LITE_MICRO_TEST(TestUnidirectionalLSTMFloat) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
      float_node_contents = tflite::testing::Create2x3x2X2FloatNodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float tolerance = 1e-6;
  tflite::testing::TestUnidirectionalLSTMFloat(kernel_eval_data, tolerance,
                                               tolerance, float_node_contents);
}

#if !defined(CMSIS_NN)
TF_LITE_MICRO_TEST(TestUnidirectionalLSTMInt8) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float hidden_state_tolerance = 1e-2;
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-2;
  tflite::testing::TestUnidirectionalLSTMInteger(
      kernel_eval_data, hidden_state_tolerance, cell_state_tolerance,
      int8_node_contents);
}
#endif

TF_LITE_MICRO_TEST(TestUnidirectionalLSTMInt16) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float hidden_state_tolerance = 1e-3;  // actually very close to 1e-4
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-2;
  tflite::testing::TestUnidirectionalLSTMInteger(
      kernel_eval_data, hidden_state_tolerance, cell_state_tolerance,
      int16_node_contents);
}
#endif  // !defined(XTENSA)
TF_LITE_MICRO_TESTS_END
