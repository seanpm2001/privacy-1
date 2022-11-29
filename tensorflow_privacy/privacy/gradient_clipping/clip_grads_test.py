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

import itertools
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_privacy.privacy.gradient_clipping import clip_grads
from tensorflow_privacy.privacy.gradient_clipping import layer_registry


# ==============================================================================
# Helper functions.
# ==============================================================================
def make_seqential_model(input_layer, num_dimensions, is_eager=False):
  """Creates a simple one-layer model with MAE loss and SGD optimizer, lr=1.0.
  """
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=(num_dimensions,)))
  model.add(input_layer)
  model.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
      loss=tf.keras.losses.MeanAbsoluteError(
          reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
      run_eagerly=is_eager)
  return model


def compute_true_gradient_norms(input_model, x_batch, y_batch):
  """Computes the real gradient norms for an input (model, x, y)."""
  loss_config = input_model.loss.get_config()
  loss_config['reduction'] = tf.keras.losses.Reduction.NONE
  per_example_loss_fn = input_model.loss.from_config(loss_config)
  with tf.GradientTape(persistent=True) as tape:
    y_pred = input_model(x_batch)
    loss = per_example_loss_fn(y_batch, y_pred)
  sqr_norms = []
  for var in input_model.trainable_variables:
    jacobian = tape.jacobian(loss, var)
    reduction_axes = range(1, len(tf.shape(jacobian)), 1)
    sqr_norms.append(tf.reduce_sum(tf.square(jacobian), axis=reduction_axes))
  sqr_norm_tsr = tf.stack(sqr_norms, axis=1)
  return tf.sqrt(tf.reduce_sum(sqr_norm_tsr, axis=1))


def wrap_test_compute_gradient_norm_on_generic_layer(test_case_obj,
                                                     layer_generator, input_dim,
                                                     output_dim):
  """Helpful testing wrapper function used to avoid code duplication.

  Args:
    test_case_obj: A tf.test.TestCase instance to run unit tests on.
    layer_generator: A function which takes in two arguments, idim and odim, and
      returns a layer that accepts input tensors of dimension idim and returns
      output tensors of dimension odim.
    input_dim: The input dimension of the test tf.keras.Model instance.
    output_dim: The output dimension of the test tf.keras.Model instance.
  """
  for e in [False, True]:
    for x_batch in get_nd_test_batches(input_dim):
      layer_instance = layer_generator(input_dim, output_dim)
      model = make_seqential_model(layer_instance, input_dim, e)
      y_pred = model(x_batch)
      y_batch = tf.ones_like(y_pred)
      computed_norms = clip_grads.compute_gradient_norms(
          model,
          x_batch,
          y_batch,
          layer_registry=layer_registry.make_default_layer_registry())
      true_norms = compute_true_gradient_norms(model, x_batch, y_batch)
      test_case_obj.assertAllClose(computed_norms, true_norms)


# ==============================================================================
# Factory functions.
# ==============================================================================
def get_nd_test_tensors(n):
  """Returns a list of candidate test for a given dimension n."""
  return [
      tf.zeros((n,), dtype=tf.float64),
      tf.convert_to_tensor(range(n), dtype_hint=tf.float64)
  ]


def get_nd_test_batches(n):
  """Returns a list of candidate input batches of dimension n."""
  result = []
  tensors = get_nd_test_tensors(n)
  for batch_size in range(1, len(tensors) + 1, 1):
    combinations = list(
        itertools.combinations(get_nd_test_tensors(n), batch_size))
    result = result + [tf.stack(ts, axis=0) for ts in combinations]
  return result


def get_test_clip_values():
  """Returns a list of candidate clip_weights."""
  return [1E-6, 0.5, 1.0, 2.0, 10.0, 1E6]


# ==============================================================================
# Main tests.
# ==============================================================================
class ClipGradsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='input_dim_1', input_dim=1),
      dict(testcase_name='input_dim_2', input_dim=2))
  def test_compute_clip_weights(self, input_dim):
    tol = 1E-6
    for t in get_nd_test_tensors(input_dim):
      self.assertIsNone(clip_grads.compute_clip_weights(None, t))
      for c in get_test_clip_values():
        weights = clip_grads.compute_clip_weights(c, t)
        self.assertAllLessEqual(t * weights, c + tol)

  @parameterized.named_parameters(
      dict(testcase_name='dense_1_1', input_dim=1, output_dim=1),
      dict(testcase_name='dense_1_2', input_dim=1, output_dim=2),
      dict(testcase_name='dense_2_1', input_dim=2, output_dim=1),
      dict(testcase_name='dense_2_2', input_dim=2, output_dim=2))
  def test_compute_gradient_norms_on_dense_layer(self, input_dim, output_dim):
    wrap_test_compute_gradient_norm_on_generic_layer(
        self, lambda idim, odim: tf.keras.layers.Dense(odim), input_dim,
        output_dim)

  @parameterized.named_parameters(
      dict(testcase_name='embedding_1_1', input_dim=1, output_dim=1),
      dict(testcase_name='embedding_1_2', input_dim=1, output_dim=2),
      dict(testcase_name='embedding_2_1', input_dim=2, output_dim=1),
      dict(testcase_name='embedding_2_2', input_dim=2, output_dim=2))
  def test_compute_gradient_norms_on_embedding_layer(self, input_dim,
                                                     output_dim):
    wrap_test_compute_gradient_norm_on_generic_layer(self,
                                                     tf.keras.layers.Embedding,
                                                     input_dim, output_dim)


if __name__ == '__main__':
  tf.test.main()
