import tensorflow as tf
from pca_wavelet_utils import build1D
import os
import matplotlib.pyplot as plt
import numpy as np

class ModelBroker():
    def __init__(self, trainset=None, testset = None, count = 3,keep_percent=1.0, dirname = "model", sample_size=100,activity_regularizer=None,inverse_activity_regularizer=None,activation_before=False,flip = False):
        self.activity_regularizer = activity_regularizer
        self.inverse_activity_regularizer = inverse_activity_regularizer
        self.activation_before = activation_before
        self.count = count
        self.keep_percent = keep_percent
        self.dirname = "models/" +dirname
        self.samplesize = sample_size
        self.trainset = trainset
        self.testset = testset
        self.flip = flip
        
    def extract_mean(self,invhead, testset):
        lastLayerIndex = -1
        sample = next(iter(testset.take(1)))[0]
        print("sample.shape",sample.shape)

        sample = sample*0.0
        lastLayer = invhead.get_layer(index=lastLayerIndex)
        mean = lastLayer([sample])[0]
        print("mean.shape",mean.shape)

        return mean
    
    def save_weights(self,head, invhead):
        
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
            print("made directory:",self.dirname)
        print("saving to:",self.dirname)
        sample = next(iter(self.testset.shuffle(100)))[0]
        sample = tf.reshape(sample, [1,sample.shape[0], sample.shape[1], sample.shape[2]])
        out = head(sample)
        sample = sample*0.0
        lastLayerIndex = -1
        lastLayer = invhead.get_layer(index=lastLayerIndex)
        mean = lastLayer(sample)
        tf.io.write_file(self.dirname + '/mean.json', tf.io.serialize_tensor(mean))
        head.save_weights(self.dirname + '/head-weights.h5')
        out = head(sample)
        print("out.shape",out.shape)
        sample = invhead(out)
        invhead.save_weights(self.dirname + '/invhead-weights.h5')
        
        
    def load_model(self):
        head, invhead = build1D(self.trainset.take(100),
                                count=self.count,
                                keep_percent = self.keep_percent,
                                samplesize=self.samplesize,
                                flip=self.flip,
                                activity_regularizer=self.activity_regularizer,
                                inverse_activity_regularizer=self.inverse_activity_regularizer,
                                activation_before=self.activation_before,
                                )
        sample = next(iter(self.testset.shuffle(100)))[0]
        print("sample.shape",sample.shape)
        sample = tf.reshape(sample, [1,sample.shape[0], sample.shape[1], sample.shape[2]])
        print("after reshape: sample.shape",sample.shape)
        print("loading from:",self.dirname)
        out = head(sample)
        head.load_weights(self.dirname + '/head-weights.h5')
        out = head(sample)
        print("out.shape",out.shape)
        sample = invhead(out)
        invhead.load_weights(self.dirname + '/invhead-weights.h5')
        mean = tf.io.parse_tensor(tf.io.read_file(self.dirname + '/mean.json'),out_type=tf.float64)
        lastLayerIndex = -1
        lastLayer = invhead.get_layer(index=lastLayerIndex)
        lastLayer.mean = mean
        firstLayer = head.get_layer(index=0)
        firstLayer.mean = -mean
        return head, invhead
        
    def build_model(self):
        head, invhead = build1D(self.trainset, 
                                count=self.count,
                                keep_percent = self.keep_percent,
                                samplesize=self.samplesize,
                                flip=self.flip,
                                activity_regularizer=self.activity_regularizer,
                                inverse_activity_regularizer=self.inverse_activity_regularizer,
                                activation_before=self.activation_before)
        self.save_weights(head, invhead)
        return head, invhead
    
    
    def check_build(self,head,invhead,testset,stats_only = False):
 
        sample = next(iter(testset.shuffle(100)))[0]
        pred = head([sample])
        recon = invhead(pred)[0]        
        psnr = 10*np.log10( 1.0 /((np.linalg.norm(recon-sample)**2)/np.prod(sample.shape)))
        ncc = np.corrcoef(tf.reshape(sample, [-1]), tf.reshape(recon, [-1]))
        if not stats_only:
            plt.subplot(221)
            plt.title('Original')
            plt.imshow(sample)
            print("sample.shape",sample.shape)
            plt.subplot(222)
            plt.title('Slice')
            plt.imshow(pred[0,:,:,0]+0.5)
            plt.subplot(223)
            plt.title('Slice')
            plt.imshow(pred[0,:,:,1]+0.5)
            print("pred.shape",pred.shape)
            print("recon.shape",recon.shape)
            plt.subplot(224)
            plt.title('Filtered')
            plt.imshow(recon)
            print("sample.dtype",sample.dtype)
            print("recon[0].dtype",recon.dtype)
            print("np.prod(sample.shape)",np.prod(sample.shape))
            print("psnr = ", psnr)
            print("ncc = ", ncc)
            print("sample[30:34,30:34,0]",sample[30:34,30:34,0])
            print("recon[30:34,30:34,0]",recon[30:34,30:34,0])
        return psnr, ncc

