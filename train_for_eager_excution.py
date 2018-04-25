# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

r"""Training executable for detection models.

This executable is used to train DetectionModels. There are two ways of
configuring the training job:

1) A single pipeline_pb2.TrainEvalPipelineConfig configuration file
can be specified by --pipeline_config_path.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files can be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being trained, an
input_reader_pb2.InputReader file to specify what training data will be used and
a train_pb2.TrainConfig file to configure training parameters.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --model_config_path=model_config.pbtxt \
        --train_config_path=train_config.pbtxt \
        --input_config_path=train_input_config.pbtxt
"""

import functools
import json
import os
import tensorflow as tf

from object_detection.my import trainer
from object_detection.builders import dataset_builder
from object_detection.my.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util
import sys
import cv2
import numpy as np
sys.path.append('../')


def get_debug_dataset():
    #one tfrecord example
    groundtruth_difficult = []
    groundtruth_group_of = []
    groundtruth_weights = [1. ,1. ,1., 1.]
    groundtruth_is_crowd = [False, False, False, False]
    key = b'd0d62a2eed6433c8d65bae1c8dca849eb678d35a8f415e4a11e5daada24eb93a'
    groundtruth_boxes = [[0.40602776, 0.5612031,  0.9992778,  0.73690623],
                         [0.06155556, 0.5310625,  0.8969167,  0.7715    ],
                         [0.48005554, 0.7369375,  0.61366665, 0.7930625 ],
                         [0.50919443, 0.75939065, 0.6063611,  0.80725   ]]
    image = cv2.imread('COCO_val2014_000000391895.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    groundtruth_area = [12190.445,   14107.271,     708.26056,   626.9852 ]
    groundtruth_classes = [4, 1, 1, 2]
    filename = b'000000391895.jpg'
    num_groundtruth_boxes = 4
    source_id = b'391895'
    
    '''
    from_tensors
    @staticmethod
    from_tensors(tensors)
    Creates a Dataset with a single element, comprising the given tensors.
    '''
    dataset = tf.data.Dataset.from_tensors({
       "groundtruth_difficult": tf.constant(groundtruth_difficult, dtype=tf.int64),
       "groundtruth_group_of": tf.constant(groundtruth_group_of, dtype=tf.int64),
       "groundtruth_weights": tf.constant(groundtruth_weights, dtype=tf.float32),
       "groundtruth_is_crowd": tf.constant(groundtruth_is_crowd, dtype=tf.bool),
       "key": tf.constant(key, dtype=tf.string),
       "groundtruth_boxes": tf.constant(groundtruth_boxes, dtype=tf.float32),
       "image": tf.constant(image, dtype=tf.float32),
       "groundtruth_area": tf.constant(groundtruth_area, dtype=tf.float32),
       "groundtruth_classes": tf.constant(groundtruth_classes, dtype=tf.int64),
       "filename": tf.constant(filename, dtype=tf.string),
       "num_groundtruth_boxes": tf.constant(num_groundtruth_boxes, dtype=tf.int32),
       "source_id": tf.constant(source_id, dtype=tf.string)
       })
    
    return dataset
  
tf.enable_eager_execution()
tf.executing_eagerly() 
import tensorflow.contrib.eager as tfe


tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.')

flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.train_dir, '`train_dir` is missing.'
  if FLAGS.task == 0: tf.gfile.MakeDirs(FLAGS.train_dir)
  if FLAGS.pipeline_config_path:
    configs = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path)
    if FLAGS.task == 0:
      tf.gfile.Copy(FLAGS.pipeline_config_path,
                    os.path.join(FLAGS.train_dir, 'pipeline.config'),
                    overwrite=True)
  else:
    configs = config_util.get_configs_from_multiple_files(
        model_config_path=FLAGS.model_config_path,
        train_config_path=FLAGS.train_config_path,
        train_input_config_path=FLAGS.input_config_path)
    if FLAGS.task == 0:
      for name, config in [('model.config', FLAGS.model_config_path),
                           ('train.config', FLAGS.train_config_path),
                           ('input.config', FLAGS.input_config_path)]:
        tf.gfile.Copy(config, os.path.join(FLAGS.train_dir, name),
                      overwrite=True)

  model_config = configs['model']
  train_config = configs['train_config']
  input_config = configs['train_input_config']

  model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=True)

  def get_next():
    datasetmy = get_debug_dataset()
#     iterator = datasetmy.make_one_shot_iterator()
#     tensors = iterator.get_next()   
#     dataset = datasetmy.batch(1)
    for batch in tfe.Iterator(datasetmy):
#          print(batch)
      return batch   
  
  create_input_dict_fn = get_next

  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  task_data = env.get('task', None) or {'type': 'master', 'index': 0}
  task_info = type('TaskSpec', (object,), task_data)

  # Parameters for a single worker.
  ps_tasks = 0
  worker_replicas = 1
  worker_job_name = 'lonely_worker'
  task = 0
  is_chief = True
  master = ''

  if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
    worker_replicas = len(cluster_data['worker']) + 1
  if cluster_data and 'ps' in cluster_data:
    ps_tasks = len(cluster_data['ps'])

  if worker_replicas > 1 and ps_tasks < 1:
    raise ValueError('At least 1 ps task is needed for distributed training.')

  if worker_replicas >= 1 and ps_tasks > 0:
    # Set up distributed training.
    server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                             job_name=task_info.type,
                             task_index=task_info.index)
    if task_info.type == 'ps':
      server.join()
      return

    worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
    task = task_info.index
    is_chief = (task_info.type == 'master')
    master = server.target

  trainer.train(create_input_dict_fn, model_fn, train_config, master, task,
                FLAGS.num_clones, worker_replicas, FLAGS.clone_on_cpu, ps_tasks,
                worker_job_name, is_chief, FLAGS.train_dir)


if __name__ == '__main__':
  tf.app.run()
