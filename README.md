# **This project helps you debug object_detection(SSD,FASTER...) using tensorflow eager excution model!**

1. The difference between train_for_eager_excution.py and origin train.py file is that the dataset object was replaced.

2. You can debug your ssd,faster,mask-faster using eager excution model as before,which is very convenient!

3. The usage of train_for_eager_excution.py is the same as the origin train.py file.

4. The inspect_tfrecord.py file helps you display the input image with bbox direct from coco tfrecord file.

5. There are another three changes in code you need to make:

6. The dataset input:  get_next function

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
       
     def get_next():
       datasetmy = get_debug_dataset()
   #     iterator = datasetmy.make_one_shot_iterator()
   #     tensors = iterator.get_next()   
   #     dataset = datasetmy.batch(1)
       for batch in tfe.Iterator(datasetmy):
   #          print(batch)
   #       return batch
         return batch['image'], batch['groundtruth_boxes'], batch['groundtruth_classes'], [], []   
   ```

   ​

7. remove outputs_collections=end_points_collection in resnet_v1.py

   ![1526350488487](C:\Users\baiyu9\AppData\Local\Temp\1526350488487.png)

   ​

   add net outputs by hand in resnet_utils.py

   ![1526350551417](C:\Users\baiyu9\AppData\Local\Temp\1526350551417.png)

8. Using tensorflow-cpu only.Because the we have to set device(gpu or cpu) by hand in eager excution model.
