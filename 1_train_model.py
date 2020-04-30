#This code ran on Tensorflow 2.0
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]='0'

from custom_model import unet, print_model_layers
import numpy as np
from imageio import imread
import random
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow import random as tfrandom
import tensorflow as tf

def datagen(batchsize=32,datalist=None,datadir=None,augment=True,pool_number=4,imagesize=(256,256)):
    while True:
        rlen=imagesize[0]
        clen=imagesize[1]

        x = np.zeros((batchsize,rlen,clen,2))
        y = np.zeros((batchsize,rlen,clen,1))

        count=0
        for ii in np.random.choice(datalist, size=batchsize):
            rstart = np.random.randint(0,imagesize[0]-rlen+1)
            cstart = np.random.randint(0,imagesize[1]-clen+1)

            datafile = os.path.join(datadir,ii)
            data = imread(datafile).astype(np.float32)

            if augment:
                #some basic data augmentation
                #rotating and flipping
                data = np.rot90(data,k=np.random.randint(0,4),axes=(0,1)) #k is 0 1 2 3
                if np.random.rand()<0.5:
                    data = np.flip(data,axis=np.random.randint(0,2)) #0 or 1

                #random scaling
                if (np.random.rand()<0.5) and (data[...,0:2].max()>0.5):
                    max0=data[...,0].max()
                    max1=data[...,1].max()
                    radius0=1-max0
                    radius1=1-max1
                    data[...,0] = data[...,0]/max0*(np.random.rand()*(2*radius0)+(max0-radius0))
                    data[...,1] = data[...,1]/max1*(np.random.rand()*(2*radius1)+(max1-radius1))

            x[count,:,:,:]=data[rstart:rstart+rlen,cstart:cstart+clen,0:2]
            y[count,:,:,0]=data[rstart:rstart+rlen,cstart:cstart+clen,2]
            count+=1
        yield (x,y)


# Massive fold loop. Keep in mind that you should have a completely held out test set for evaluation.
seedval = 1337
random.seed(seedval)
np.random.seed(seedval)
tfrandom.set_seed(seedval)

trainvaltestdir = 'train_data'
trainvaltestlist = os.listdir(trainvaltestdir)
random.shuffle(trainvaltestlist)

numfolds = 5
valfrac = 1 / numfolds
numtestval = int(np.floor(len(trainvaltestlist) * valfrac))

for ii in range(numfolds):
    pool_number = 4
    model = unet(data_shape=(256, 256),
                 channels_in=2,
                 channels_out=1,
                 starting_filter_number=32,
                 kernel_size=(3, 3),
                 num_conv_per_pool=2,
                 num_repeat_bottom_conv=0,
                 pool_number=pool_number,
                 pool_size=(2, 2),
                 expansion_rate=2,
                 dropout_type='block',
                 dropout_rate=0.1,
                 dropout_power=1 / 4,
                 dropblock_size=3,
                 add_conv_layers=0,
                 add_conv_filter_number=32,
                 add_conv_dropout_rate=None,
                 final_activation='sigmoid',
                 gn_type='groups',
                 gn_param=32,
                 weight_constraint=None)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy')

    print('Valid:', ii * numtestval, (ii + 1) * numtestval)
    trainlist = trainvaltestlist[:ii * numtestval] + trainvaltestlist[(ii + 1) * numtestval:]
    vallist = trainvaltestlist[ii * numtestval:(ii + 1) * numtestval]
    print('list lengths', len(trainlist), len(vallist))

    batchsize = 4
    ckpt = ModelCheckpoint('results/best_weights_' + str(ii) + '.hdf5', monitor='val_loss', save_best_only=True,
                           save_weights_only=True)
    history = History()
    callbacks = [ckpt, history]

    train_data = datagen(batchsize=batchsize, datalist=trainlist, datadir=trainvaltestdir, augment=True,
                         pool_number=pool_number)

    val_data = datagen(batchsize=batchsize, datalist=vallist, datadir=trainvaltestdir, augment=False,
                       pool_number=pool_number)

    model.fit(train_data, epochs=2000, steps_per_epoch=100, validation_data=val_data, validation_steps=20,
              callbacks=callbacks)
    model.save_weights('results/latest_weights_' + str(ii) + '.hdf5')
    np.save('results/loss_' + str(ii) + '.npy', history.history)