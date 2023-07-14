# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This file includes kernel test targets only.

# These tests needs additional dependencies beyond libtensorflow-microlite.a.
$(eval $(call microlite_test,kernel_detection_postprocess_test,\
  $(TENSORFLOW_ROOT)detection_postprocess_test.cc \
  $(TENSORFLOW_ROOT)detection_postprocess_flexbuffers_generated_data.cc,\
  $(TENSORFLOW_ROOT)detection_postprocess_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_circular_buffer_test,\
  $(TENSORFLOW_ROOT)circular_buffer_test.cc \
  $(TENSORFLOW_ROOT)circular_buffer_flexbuffers_generated_data.cc,\
  $(TENSORFLOW_ROOT)circular_buffer_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_conv_test,\
  $(TENSORFLOW_ROOT)conv_test.cc \
  $(TENSORFLOW_ROOT)conv_test_common.cc \
  $(TENSORFLOW_ROOT)testdata/conv_test_data.cc,\
  $(TENSORFLOW_ROOT)conv_test.h \
  $(TENSORFLOW_ROOT)testdata/conv_test_data.h))

$(eval $(call microlite_test,kernel_transpose_conv_test,\
  $(TENSORFLOW_ROOT)transpose_conv_test.cc \
  $(TENSORFLOW_ROOT)conv_test_common.cc,\
  $(TENSORFLOW_ROOT)conv_test.h))

$(eval $(call microlite_test,kernel_lstm_eval_test,\
  $(TENSORFLOW_ROOT)lstm_eval_test.cc \
  $(TENSORFLOW_ROOT)testdata/lstm_test_data.cc,\
  $(TENSORFLOW_ROOT)testdata/lstm_test_data.h))

$(eval $(call microlite_test,unidirectional_sequence_lstm_test,\
  $(TENSORFLOW_ROOT)unidirectional_sequence_lstm_test.cc \
  $(TENSORFLOW_ROOT)testdata/lstm_test_data.cc,\
  $(TENSORFLOW_ROOT)testdata/lstm_test_data.h))

# For kernel tests without extra dependencies (beyond libtensorflow-microlite.a),
# use simple for loop to generate their make targets in a common way.
MICROLITE_KERNEL_SIMPLE_TEST_SRCS := \
$(TENSORFLOW_ROOT)activations_test.cc \
$(TENSORFLOW_ROOT)add_test.cc \
$(TENSORFLOW_ROOT)add_n_test.cc \
$(TENSORFLOW_ROOT)arg_min_max_test.cc \
$(TENSORFLOW_ROOT)batch_to_space_nd_test.cc \
$(TENSORFLOW_ROOT)broadcast_args_test.cc \
$(TENSORFLOW_ROOT)broadcast_to_test.cc \
$(TENSORFLOW_ROOT)cast_test.cc \
$(TENSORFLOW_ROOT)ceil_test.cc \
$(TENSORFLOW_ROOT)comparisons_test.cc \
$(TENSORFLOW_ROOT)concatenation_test.cc \
$(TENSORFLOW_ROOT)cumsum_test.cc \
$(TENSORFLOW_ROOT)depth_to_space_test.cc \
$(TENSORFLOW_ROOT)depthwise_conv_test.cc \
$(TENSORFLOW_ROOT)dequantize_test.cc \
$(TENSORFLOW_ROOT)div_test.cc \
$(TENSORFLOW_ROOT)elementwise_test.cc \
$(TENSORFLOW_ROOT)elu_test.cc \
$(TENSORFLOW_ROOT)embedding_lookup_test.cc \
$(TENSORFLOW_ROOT)exp_test.cc \
$(TENSORFLOW_ROOT)expand_dims_test.cc \
$(TENSORFLOW_ROOT)fill_test.cc \
$(TENSORFLOW_ROOT)floor_test.cc \
$(TENSORFLOW_ROOT)floor_div_test.cc \
$(TENSORFLOW_ROOT)floor_mod_test.cc \
$(TENSORFLOW_ROOT)fully_connected_test.cc \
$(TENSORFLOW_ROOT)gather_test.cc \
$(TENSORFLOW_ROOT)gather_nd_test.cc \
$(TENSORFLOW_ROOT)hard_swish_test.cc \
$(TENSORFLOW_ROOT)l2norm_test.cc \
$(TENSORFLOW_ROOT)l2_pool_2d_test.cc \
$(TENSORFLOW_ROOT)leaky_relu_test.cc \
$(TENSORFLOW_ROOT)logical_test.cc \
$(TENSORFLOW_ROOT)logistic_test.cc \
$(TENSORFLOW_ROOT)log_softmax_test.cc \
$(TENSORFLOW_ROOT)maximum_minimum_test.cc \
$(TENSORFLOW_ROOT)mirror_pad_test.cc \
$(TENSORFLOW_ROOT)mul_test.cc \
$(TENSORFLOW_ROOT)neg_test.cc \
$(TENSORFLOW_ROOT)pack_test.cc \
$(TENSORFLOW_ROOT)pad_test.cc \
$(TENSORFLOW_ROOT)pooling_test.cc \
$(TENSORFLOW_ROOT)prelu_test.cc \
$(TENSORFLOW_ROOT)quantization_util_test.cc \
$(TENSORFLOW_ROOT)quantize_test.cc \
$(TENSORFLOW_ROOT)reduce_test.cc \
$(TENSORFLOW_ROOT)reshape_test.cc \
$(TENSORFLOW_ROOT)resize_bilinear_test.cc \
$(TENSORFLOW_ROOT)resize_nearest_neighbor_test.cc \
$(TENSORFLOW_ROOT)round_test.cc \
$(TENSORFLOW_ROOT)select_test.cc \
$(TENSORFLOW_ROOT)shape_test.cc \
$(TENSORFLOW_ROOT)slice_test.cc \
$(TENSORFLOW_ROOT)softmax_test.cc \
$(TENSORFLOW_ROOT)space_to_batch_nd_test.cc \
$(TENSORFLOW_ROOT)space_to_depth_test.cc \
$(TENSORFLOW_ROOT)split_test.cc \
$(TENSORFLOW_ROOT)split_v_test.cc \
$(TENSORFLOW_ROOT)squared_difference_test.cc \
$(TENSORFLOW_ROOT)squeeze_test.cc \
$(TENSORFLOW_ROOT)strided_slice_test.cc \
$(TENSORFLOW_ROOT)sub_test.cc \
$(TENSORFLOW_ROOT)svdf_test.cc \
$(TENSORFLOW_ROOT)tanh_test.cc \
$(TENSORFLOW_ROOT)transpose_test.cc \
$(TENSORFLOW_ROOT)unpack_test.cc \
$(TENSORFLOW_ROOT)while_test.cc \
$(TENSORFLOW_ROOT)zeros_like_test.cc

# Generate simple kernel test targets in a common way
$(foreach TEST_TARGET,$(MICROLITE_KERNEL_SIMPLE_TEST_SRCS),\
$(eval $(call microlite_test,kernel_$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))
