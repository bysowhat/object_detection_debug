# **This project helps you debug object_detection(SSD,FASTER...) using tensorflow eager excution model!**

1. The difference between train_for_eager_excution.py and origin train.py file is that the dataset object was replaced.

2. You can debug your ssd,faster,mask-faster using eager excution model as before,which is very convenient!

3. The usage of train_for_eager_excution.py is the same as the origin train.py file.

4. The inspect_tfrecord.py file helps you display the input image with bbox direct from coco tfrecord file.

5. There are another three changes in code you need to make:

6. **The dataset input:  get_next function**

   ```
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
       image = np.expand_dims(image, 0)
       groundtruth_area = [12190.445,   14107.271,     708.26056,   626.9852 ]
       groundtruth_classes = [[4, 1, 1, 2]]
       nb_classes = 90
       targets = np.reshape(groundtruth_classes, -1)
       groundtruth_classes = np.eye(nb_classes)[targets]
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
       
     def get_next():
       datasetmy = get_debug_dataset()
   #     iterator = datasetmy.make_one_shot_iterator()
   #     tensors = iterator.get_next()   
   #     dataset = datasetmy.batch(1)
       for batch in tfe.Iterator(datasetmy):
   #          print(batch)
   #       return batch
         return batch['image'], [batch['groundtruth_boxes']], [batch['groundtruth_classes']], [], []   
   ```

   ​

7. **remove outputs_collections**=end_points_collection in resnet_v1.py resnet_v1fun

   Because outputs_collections using the attribute of 'alias' in tensor,however,the tensor has no 'alias' in eager excution model.

   ```
   with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
       with slim.arg_scope([slim.conv2d, bottleneck,
                            resnet_utils.stack_blocks_dense]):
         with slim.arg_scope([slim.batch_norm], is_training=is_training):
           net = inputs
           if include_root_block:
             if output_stride is not None:
               if output_stride % 4 != 0:
                 raise ValueError('The output_stride needs to be a multiple of 4.')
               output_stride /= 4
             net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
             net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
           net, end_points = resnet_utils.stack_blocks_dense(net, blocks, output_stride,
                                                 store_non_strided_activations)
   ```

   ​

   **add net outputs manually ** in resnet_utils.py stack_blocks_dense fun

   ```
   # The atrous convolution rate parameter.
     rate = 1

     end_points = {}
     
     for block in blocks:
       with tf.variable_scope(block.scope, 'block', [net]) as sc:
         block_stride = 1
         for i, unit in enumerate(block.args):
           if store_non_strided_activations and i == len(block.args) - 1:
             # Move stride from the block's last unit to the end of the block.
             block_stride = unit.get('stride', 1)
             unit = dict(unit, stride=1)

           with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
             # If we have reached the target output_stride, then we need to employ
             # atrous convolution with stride=1 and multiply the atrous rate by the
             # current unit's stride for use in subsequent layers.
             if output_stride is not None and current_stride == output_stride:
               net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
               rate *= unit.get('stride', 1)

             else:
               net = block.unit_fn(net, rate=1, **unit)
               current_stride *= unit.get('stride', 1)
               if output_stride is not None and current_stride > output_stride:
                 raise ValueError('The target output_stride cannot be reached.')

         end_points[block.scope] = net
   ```

   ​

8. **Using tensorflow-cpu only**.Because the we have to set device(gpu or cpu) manually in eager excution model.

9. **If debugging rfcn**, these changs should also be made in core/post_processing.py/batch_multiclass_non_max_suppression fun

   ```
    #for eager excution start
         masks_shape = tf.stack([batch_size, num_anchors, 1, 1, 1])
         per_image_masks = tf.zeros(masks_shape)
         per_image_masks = tf.reshape(per_image_masks,
             [-1, q, per_image_masks.shape[2].value,
              per_image_masks.shape[3].value])
         #for eager excution end
         
   #       per_image_masks = tf.reshape(
   #           tf.slice(per_image_masks, 4 * [0],
   #                    tf.stack([per_image_num_valid_boxes, -1, -1, -1])),
   #           [-1, q, per_image_masks.shape[2].value,
   #            per_image_masks.shape[3].value])
   ```

   ​
