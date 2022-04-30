import tensorflow as tf
import tensorflow_datasets as tfds

tf.keras.backend.set_floatx('float64')

class DataLoader():
    def __init__(self,IMAGE_SIZE=64,take=None,dataset="birds"):
        if dataset == "birds":
            load_dataset = "caltech_birds2010"
        elif dataset == "pets":
            load_dataset = "oxford_iiit_pet"
        self.dataset = dataset
        self.ds = self._import_data(load_dataset)
        if take != None:
            self.ds = self.ds.take(take)
        self.IMAGE_SIZE = IMAGE_SIZE
        
    def _scale_and_resize(self,image):
        image = image / 255
        image = tf.image.resize(image, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        image = tf.cast(image, tf.float64)  
        return image

    def pre_process_segmentation(self,record):
        mask = record["segmentation_mask"]
        image = record["image"]
        image = tf.clip_by_value(image,10,255)
        combined_mask = image-image*(mask%2)
        combined_mask = self._scale_and_resize(combined_mask)
        return combined_mask

    def pre_process_image(self,record):
        image = record["image"]
        #image = tf.clip_by_value(image,10,255)
        image = self._scale_and_resize(image)
        return image

    def _import_data(self,dataset):
        builder = tfds.builder(dataset)
        builder.download_and_prepare()
        ds = builder.as_dataset(split=["all"])[0]
        return ds

    def import_processed_img(self):
        img_ds = self.ds.map(lambda x:[self.pre_process_image(x)])
        return img_ds

    def import_processed_seg(self):
        seg_ds = self.ds.map(lambda x:[self.pre_process_segmentation(x)])
        return seg_ds
    
