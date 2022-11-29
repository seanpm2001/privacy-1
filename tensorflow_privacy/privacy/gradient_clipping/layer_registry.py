# Copyright 2022, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates a default layer registry.

Defines "fast" gradient norm layer registry functions that output a triple
(output, sqr_grad_norms, vars), where output is the pre-activator tensor,
sqr_grad_norms is the square of the norm of the layer's input, and vars
is an ordered list of the trainable weights.

These registry functions are registered in a 'registry' (dictionary) whose key
is the layer class and whose value is the registry function.
"""

import tensorflow as tf


# ==============================================================================
# Supported Keras layers
# ==============================================================================
def dense_layer_computation(layer_instance, inputs, tape):
  """Registry function for tf.keras.layers.Dense."""
  kernel_index = 0
  bias_index = 1
  reg_vars = tf.matmul(*inputs, layer_instance.trainable_weights[kernel_index])
  tape.watch(reg_vars)
  if len(layer_instance.trainable_weights) > 1:  # add bias term
    reg_vars += layer_instance.trainable_weights[bias_index]
  sqr_grad_norms = tf.reduce_sum(tf.square(*inputs), axis=1)
  if len(layer_instance.trainable_weights) > 1:
    sqr_grad_norms += tf.cast(1.0, dtype=sqr_grad_norms.dtype)
  outputs = layer_instance.activation(reg_vars)
  return outputs, [sqr_grad_norms], [reg_vars]


def embedding_layer_computation(layer_instance, inputs, tape):
  """Registry function for tf.keras.layers.Embedding."""
  tape.watch(layer_instance.trainable_weights)
  sqr_grad_norms = tf.ones((tf.shape(*inputs)[0],))
  outputs = layer_instance(*inputs)
  return outputs, [sqr_grad_norms], [outputs]


# ==============================================================================
# Main factory methods
# ==============================================================================
def make_default_layer_registry():
  registry = {}
  registry[hash(tf.keras.layers.Dense)] = dense_layer_computation
  registry[hash(tf.keras.layers.Embedding)] = embedding_layer_computation
  return registry
