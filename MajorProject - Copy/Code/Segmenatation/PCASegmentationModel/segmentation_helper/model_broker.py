import tensorflow as tf
from pca_wavelet_utils import build1D
import os
import matplotlib.pyplot as plt
import numpy as np

class ModelBroker():
    def __init__(self):
        pass
    
    def extract_mean(self,invhead, testset):
        lastLayerIndex = -1
        sample = next(iter(testset.take(1)))[0]
        print("sample.shape",sample.shape)

        sample = sample*0.0
        lastLayer = invhead.get_layer(index=lastLayerIndex)
        mean = lastLayer([sample])[0]
        print("mean.shape",mean.shape)

        return mean
    
    def save_weights(self,head, invhead, testset, dirname):
        dirname = "models/" +dirname
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print("made directory:",dirname)
        print("saving to:",dirname)
        sample = next(iter(testset.shuffle(100)))[0]
        sample = tf.reshape(sample, [1,sample.shape[0], sample.shape[1], sample.shape[2]])
        out = head(sample)
        sample = sample*0.0
        lastLayerIndex = -1
        lastLayer = invhead.get_layer(index=lastLayerIndex)
        mean = lastLayer(sample)
        tf.io.write_file(dirname + '/mean.json', tf.io.serialize_tensor(mean))
        head.save_weights(dirname + '/head-weights.h5')
        out = head(sample)
        print("out.shape",out.shape)
        sample = invhead(out)
        invhead.save_weights(dirname + '/invhead-weights.h5')
        
        
    def load_model(self, trainset, testset,dirname="model",keep_percent=1.0,count=3):
        head, invhead = build1D(trainset.take(100), count=count, keep_percent = keep_percent,samplesize=100, flip=False)
        sample = next(iter(testset.shuffle(100)))[0]
        print("sample.shape",sample.shape)
        sample = tf.reshape(sample, [1,sample.shape[0], sample.shape[1], sample.shape[2]])
        print("after reshape: sample.shape",sample.shape)
        dirname = "models/" + dirname
        print("loading from:",dirname)
        out = head(sample)
        head.load_weights(dirname + '/head-weights.h5')
        out = head(sample)
        print("out.shape",out.shape)
        sample = invhead(out)
        invhead.load_weights(dirname + '/invhead-weights.h5')
        mean = tf.io.parse_tensor(tf.io.read_file(dirname + '/mean.json'),out_type=tf.float64)
        lastLayerIndex = -1
        lastLayer = invhead.get_layer(index=lastLayerIndex)
        lastLayer.mean = mean
        firstLayer = head.get_layer(index=0)
        firstLayer.mean = -mean
        return head, invhead
        
    def build_model(self,trainset,testset,count=3,keep_percent=1.0, samplesize=100,dirname="model"):
        head, invhead = build1D(trainset, count=count,keep_percent = keep_percent,samplesize=samplesize, flip=False)
        self.save_weights(head, invhead, testset, dirname)
        return head, invhead
    
    
    def check_build(self,head,invhead,testset):
        plt.subplot(221)
        plt.title('Original')
        sample = next(iter(testset.shuffle(100)))[0]

        plt.imshow(sample)
        print("sample.shape",sample.shape)

        pred = head([sample])

        plt.subplot(222)
        plt.title('Slice')
        plt.imshow(pred[0,:,:,0]+0.5)
        plt.subplot(223)
        plt.title('Slice')
        plt.imshow(pred[0,:,:,1]+0.5)

        print("pred.shape",pred.shape)
        recon = invhead(pred)[0]
        print("recon.shape",recon.shape)
        plt.subplot(224)
        plt.title('Filtered')
        plt.imshow(recon)
        print("sample.dtype",sample.dtype)
        print("recon[0].dtype",recon.dtype)
        print("np.prod(sample.shape)",np.prod(sample.shape))
        psnr = 10*np.log10( 1.0 /((np.linalg.norm(recon-sample)**2)/np.prod(sample.shape)))
        ncc = np.corrcoef(tf.reshape(sample, [-1]), tf.reshape(recon, [-1]))
        print("psnr = ", psnr)
        print("ncc = ", ncc)
        print("sample[30:34,30:34,0]",sample[30:34,30:34,0])
        print("recon[30:34,30:34,0]",recon[30:34,30:34,0])

