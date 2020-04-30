import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from model.custom_model import unet
from evaluation_functions import *
import numpy as np
import matplotlib.pyplot as plt

#change the foldnum here to access different cross validation models
foldnum=0

model = unet(data_shape=(256,256),
         channels_in=2,
         channels_out=1,
         starting_filter_number=32,
         kernel_size=(3,3),#[(3,3),(3,3),(3,3),(5,5)],
         num_conv_per_pool=2,
         num_repeat_bottom_conv=0,
         pool_number=4,
         pool_size=(2,2),
         expansion_rate=2,
         dropout_type='block',
         dropout_rate=0.1,
         dropout_power=1/4,
         dropblock_size=3,
         add_conv_layers=0,
         add_conv_filter_number=32,
         add_conv_dropout_rate=None,
         final_activation='sigmoid',
         gn_type='groups',
         gn_param=32,
         weight_constraint=None)

model.load_weights('results/fold_+'+str(foldnum)+'+/best_weights.hdf5')


nt = 1000
thresholdlist = [ii / nt for ii in range(nt + 1)]

gtp = np.zeros(shape=(len(thresholdlist),))
gtn = np.zeros(shape=(len(thresholdlist),))
gfp = np.zeros(shape=(len(thresholdlist),))
gfn = np.zeros(shape=(len(thresholdlist),))

ltp = []
ltn = []
lfp = []
lfn = []

datafolder = 'test_data'
for datafile in os.listdir(datafolder):
    print('Accessing', datafile)
    data = read_image_data(os.path.join(datafolder, datafile))

    print('Predicting myocardiocytes')
    pmask = predict_mc(data, model)

    print('Getting myocardiocyte mask')
    tmask = patchwise_mc_nuclei_masks(data, nuclei='mc')

    print('Getting nuclei')
    nmask = patchwise_mc_nuclei_masks(data, nuclei='all')

    print('Analyzing Data')

    tp, tn, fp, fn, threshold = analyze_data_onthefly(data, tmask, pmask, nmask, thresholdlist=thresholdlist)

    gtp += tp
    gtn += tn
    gfp += fp
    gfn += fn

    ltp.append(tp)
    ltn.append(tn)
    lfp.append(fp)
    lfn.append(fn)

    sens, spec = calc_sens_spec(tp, tn, fp, fn)

    plt.figure(figsize=(8, 8))
    plt.plot(1 - spec, sens, linewidth=4)
    plt.plot(np.arange(0, 101) / 100, np.arange(0, 101) / 100, color='k', linestyle='--')
    plt.xlabel('1-specificity', fontsize=20)
    plt.ylabel('sensitivity', fontsize=20)
    plt.title(datafile + '\nAUC = ' + str(np.sum(np.diff(spec) * (sens[:-1] + sens[1:]) / 2)), fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.show()

sens, spec = calc_sens_spec(gtp, gtn, gfp, gfn)

plt.figure(figsize=(8, 8))
plt.plot(1 - spec, sens, linewidth=4)
plt.plot(np.arange(0, 101) / 100, np.arange(0, 101) / 100, color='k', linestyle='--')
plt.xlabel('1-specificity', fontsize=20)
plt.ylabel('sensitivity', fontsize=20)
plt.title('All Data\nAUC = ' + str(np.sum(np.diff(spec) * (sens[:-1] + sens[1:]) / 2)), fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()