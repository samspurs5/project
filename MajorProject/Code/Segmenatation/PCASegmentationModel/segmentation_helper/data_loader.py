import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tf.keras.backend.set_floatx('float64')

class DataLoader():
    def __init__(self,IMAGE_SIZE=64,debug=False):
        self.ds_train, self.ds_test, _ = self._import_data()
        self.debug = debug
        if debug:
            self.ds_train = self.ds_train.take(101)
            self.ds_test = self.ds_test.take(101)
        self.IMAGE_SIZE = IMAGE_SIZE
        self.combined_train = self.ds_train.map(lambda x:[self.pre_process_dataset(x)])
        self.combined_test = self.ds_test.map(lambda x:[self.pre_process_dataset(x)])
        
            
        
    def combine(self,record):
        image = record[0]["image"]
        mask = record[0]["segmentation_mask"]
        image = np.ones(mask.shape)*mask + image * abs(mask-np.ones(mask.shape))
        return [image]

    def combine_dataset(self,record):
        new_item = [] 
        image = record[0]["image"]
        mask = record[0]["segmentation_mask"]
        new_item.append(image)
        new_item.append(np.ones(mask.shape)*mask + image * abs(mask-np.ones(mask.shape)))
        return [new_item]


    def pre_process_image(self,record,key="image"):
        image = record[key]
        if key == "image":
            image = image / 255
        image = tf.image.resize(image, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        image = tf.cast(image, tf.float64)    
        return image



    def pre_process_dataset(self,record,key=["image","segmentation_mask"]):
        new_item = {}
        for k in key:
            image = record[k]
            if k == "image":
                image = image / 255
            image = tf.image.resize(image, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            new_item[k] = tf.cast(image, tf.float64)    
        return new_item

    def _import_data(self):
        ds_train,info = tfds.load('caltech_birds2010', split='train',with_info=True,)
        ds_test = tfds.load('caltech_birds2010', split='test')
        return ds_train,ds_test,info


    def import_processed_img(self):
        img_trainset = self.ds_train.map(lambda x:[self.pre_process_image(x)])
        img_testset = self.ds_test.map(lambda x:[self.pre_process_image(x)])
        return img_trainset,img_testset

    def import_processed_seg(self):
        seg_trainset = tf.data.Dataset.from_tensor_slices(list(map(self.combine,
                                                                   tfds.as_numpy(self.combined_train))))
        seg_testset = tf.data.Dataset.from_tensor_slices(list(map(self.combine,
                                                                  tfds.as_numpy(self.combined_test))))
        return seg_trainset,seg_testset
    
    def import_processed_combined(self):
        combined_train = tf.data.Dataset.from_tensor_slices(list(map(self.combine_dataset,
                                                                     tfds.as_numpy(self.combined_train))))
        combined_test = tf.data.Dataset.from_tensor_slices(list(map(self.combine_dataset,
                                                                    tfds.as_numpy(self.combined_test))))
        
        return combined_train,combined_test
    
    def import_raw_combined(self):
        return self.combined_train, self.combined_test
        