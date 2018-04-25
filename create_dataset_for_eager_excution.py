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
from object_detection.utils import label_map_util
from object_detection.builders import preprocessor_builder
from object_detection.utils import visualization_utils as vis_util
import numpy as np
from matplotlib import pyplot as plt
import cv2

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
    image = np.expand_dims(image, 0)
    groundtruth_area = [12190.445,   14107.271,     708.26056,   626.9852 ]
    groundtruth_classes = [4, 1, 1, 2]
    filename = b'000000391895.jpg'
    num_groundtruth_boxes = 4
    source_id = b'391895'
    
#     dataset = tf.data.TextLineDataset('debugdataset.txt')
#     dataset = tf.data.Dataset.from_tensor_slices({
#       "groundtruth_difficult": np.array(groundtruth_difficult),
#       "groundtruth_group_of": np.array(groundtruth_group_of),
#       "groundtruth_weights": np.array(groundtruth_weights),
#       "groundtruth_is_crowd": np.array(groundtruth_is_crowd),
# #       "key": key,
#       "groundtruth_boxes": np.array(groundtruth_boxes),
#       "image": np.array(image),
#       "groundtruth_area": np.array(groundtruth_area),
#       "groundtruth_classes": np.array(groundtruth_classes),
#       "filename": filename,
#       "num_groundtruth_boxes": num_groundtruth_boxes,
#       "source_id": source_id
#       })

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
  
  
  
  
#   iterator = dataset_util.make_initializable_iterator(dataset_builder.build(input_config))
#   datasetmy = dataset_builder.build(input_config)
  datasetmy = get_debug_dataset()
  print(datasetmy.output_shapes)
  print(datasetmy.output_types)
  iterator = datasetmy.make_one_shot_iterator()
  
  tensors = iterator.get_next()

  #print all tensors in tfrecord
  print(tensors)
  
  groundtruth_difficult = tensors['groundtruth_difficult']
  groundtruth_group_of = tensors['groundtruth_group_of']
  groundtruth_weights = tensors['groundtruth_weights']
  groundtruth_is_crowd = tensors['groundtruth_is_crowd']
  key = tensors['key']
  groundtruth_boxes = tensors['groundtruth_boxes']
  image = tensors['image']
  groundtruth_area = tensors['groundtruth_area']
  groundtruth_classes = tensors['groundtruth_classes']
  filename = tensors['filename']
  num_groundtruth_boxes = tensors['num_groundtruth_boxes']
  source_id = tensors['source_id']
  
  
  
   
  init_op=tf.initialize_all_variables()
  with tf.Session() as sess:
#     sess.run(iterator.initializer)
    sess.run(tf.tables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(init_op)
    for i in range(10):
      groundtruth_weights_val,groundtruth_difficult_val,groundtruth_group_of_val,groundtruth_is_crowd_val,key_val,groundtruth_boxes_val,image_val,groundtruth_area_val,groundtruth_classes_val,filename_val,num_groundtruth_boxes_val,source_id_val = \
      sess.run([groundtruth_weights,groundtruth_difficult,groundtruth_group_of,groundtruth_is_crowd,key,groundtruth_boxes,image,groundtruth_area,groundtruth_classes,filename,num_groundtruth_boxes,source_id])
      print(groundtruth_weights_val)
      print(groundtruth_boxes_val)
      print(groundtruth_difficult_val)
      print(groundtruth_group_of_val)
      print(groundtruth_is_crowd_val)
      print(key_val)
      print(image_val)
      print(groundtruth_area_val)
      print(groundtruth_classes_val)
      print(filename_val)
      print(num_groundtruth_boxes_val)
      print(source_id_val)
      image_val = image_val[0]
      image_val = image_val.astype(np.uint8)
#       cv2.imshow('image', image_val)
#       cv2.waitKey()
#       plt.imshow(image_val)
#       plt.show()  
      print('finish')
      
      #plot bbox on image
      plt.switch_backend("TkAgg")
      classes_val = groundtruth_classes_val
      boxes_val = groundtruth_boxes_val
      scores_val = [1.0]*num_groundtruth_boxes_val
      image_np = image_val
      image_np_origin = image_val.copy()
      NUM_CLASSES = 90
      IMAGE_SIZE = (12, 8)
      PATH_TO_LABELS = '../../data/mscoco_label_map.pbtxt'
      label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
      categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                  use_display_name=True)
      category_index = label_map_util.create_category_index(categories)
      vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes_val,
                np.squeeze(classes_val).astype(np.int32),
                np.squeeze(scores_val),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.subplot(121)
      plt.imshow(image_np)
      plt.subplot(122)
      plt.imshow(image_np_origin)
      plt.show()  
      print('finish')
           
           
      pass
  coord.request_stop()
  coord.join(threads)





if __name__ == '__main__':
  tf.app.run()
