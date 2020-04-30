import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Dropout, SpatialDropout1D, SpatialDropout2D, SpatialDropout3D
from tensorflow.keras.layers import Conv1D, UpSampling1D, MaxPool1D
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPool2D
from tensorflow.keras.layers import Conv3D, UpSampling3D, MaxPool3D

from tensorflow.keras.constraints import UnitNorm, MaxNorm

from groupnorm import GroupNormalization
from dropblock import DB1D, DB2D, DB3D
from weightconstraint import WeightStandardization

def print_model_layers(model):
    for i, layer in enumerate(model.layers):
        print("Layer", i, "\t", layer.name, "\t\t", layer.input_shape, "\t", layer.output_shape)

def unet(data_shape=(None,None,None),
         channels_in=1,
         channels_out=1,
         starting_filter_number=32,
         kernel_size=(3,3,3),
         num_conv_per_pool=2,
         num_repeat_bottom_conv=0,
         pool_number=4,
         pool_size=(2,2,2),
         expansion_rate=2,
         dropout_type='standard',
         dropout_rate=0.25,
         dropout_power=1/4,
         dropblock_size=5,
         add_conv_layers=4,
         add_conv_filter_number=32,
         add_conv_dropout_rate=None,
         final_activation='linear',
         gn_type='groups',
         gn_param=32,
         weight_constraint=None):

    if len(data_shape)==1:
        print('using 1D operations')
        Conv=Conv1D
        UpSampling=UpSampling1D
        MaxPool=MaxPool1D
        if dropout_type == 'spatial':
            DRPT = SpatialDropout1D
            print('using spatial dropout')
        elif (dropout_type == 'dropblock') or (dropout_type == 'block'):
            DRPT = DB1D(block_size=dropblock_size)
            print('using dropblock with blocksize:', dropblock_size)
        else:
            DRPT = Dropout
            print('using standard dropout')
    elif len(data_shape)==2:
        print('using 2D operations')
        Conv=Conv2D
        UpSampling=UpSampling2D
        MaxPool=MaxPool2D
        if dropout_type == 'spatial':
            DRPT = SpatialDropout2D
            print('using spatial dropout')
        elif (dropout_type == 'dropblock') or (dropout_type == 'block'):
            DRPT = DB2D(block_size=dropblock_size)
            print('using dropblock with blocksize:', dropblock_size)
        else:
            DRPT = Dropout
            print('using standard dropout')
    elif len(data_shape)==3:
        print('using 3D operations')
        Conv=Conv3D
        UpSampling=UpSampling3D
        MaxPool=MaxPool3D
        if dropout_type == 'spatial':
            DRPT = SpatialDropout3D
            print('using spatial dropout')
        elif (dropout_type == 'dropblock') or (dropout_type == 'block'):
            DRPT = DB3D(block_size=dropblock_size)
            print('using dropblock with blocksize:', dropblock_size)
        else:
            DRPT = Dropout
            print('using standard dropout')
    else:
        print('Error: data_shape not compatible')
        return None

    if (weight_constraint=='ws') or (weight_constraint=='weightstandardization'):
        print('using weight standardization')
        wsconstraint = WeightStandardization(mean=0,std=1)
    elif (weight_constraint=='maxnorm') or (weight_constraint=='MaxNorm'):
        print('using MaxNorm')
        wsconstraint = MaxNorm(max_value=1,axis=[ii for ii in range(len(data_shape)+1)])
    elif (weight_constraint=='unitnorm') or (weight_constraint=='UnitNorm'):
        print('using UnitNorm')
        wsconstraint = UnitNorm(axis=[0,1,2])
    else:
        print('excluding weight constraints')
        wsconstraint=None

    layer_conv={}
    layer_nonconv={}

    number_of_layers_half = pool_number + 1

    number_of_filters_max = np.round((expansion_rate**(number_of_layers_half-1))*starting_filter_number)

    #first half of U
    layer_nonconv[0] = Input(data_shape+(channels_in,))
    print()
    print('Input:', layer_nonconv[0].shape)
    print()

    for layer_number in range(1,number_of_layers_half):
        number_of_filters_current = np.round((expansion_rate ** (layer_number - 1)) * starting_filter_number)
        drop_rate_layer = dropout_rate * (np.power((number_of_filters_current / number_of_filters_max),dropout_power))

        if isinstance(pool_size,(list,)):
            poolsize=pool_size[layer_number-1]
        else:
            poolsize=pool_size

        if isinstance(kernel_size,(list,)):
            kernelsize=kernel_size[layer_number-1]
        else:
            kernelsize=kernel_size

        if gn_type=='channels':
            groups = int(np.clip(number_of_filters_current/gn_param,1,number_of_filters_current))
        else:
            groups = int(np.clip(gn_param,1,number_of_filters_current))

        layer_conv[layer_number] = DRPT(rate=drop_rate_layer)(GroupNormalization(groups=groups)(Conv(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu', kernel_constraint=wsconstraint)(layer_nonconv[layer_number-1])))
        for _ in range(1,num_conv_per_pool):
            layer_conv[layer_number] = DRPT(rate=drop_rate_layer)(GroupNormalization(groups=groups)(Conv(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu', kernel_constraint=wsconstraint)(layer_conv[layer_number])))
        print('{:<30}'.format(str(layer_conv[layer_number].shape)),'\tgroups:', groups, '\tkernel:', kernelsize, '\tdroprate:',drop_rate_layer)
        layer_nonconv[layer_number] = Concatenate(axis=-1)([MaxPool(pool_size=poolsize)(layer_conv[layer_number]),DRPT(rate=drop_rate_layer)(GroupNormalization(groups=groups)(Conv(filters=number_of_filters_current,kernel_size=poolsize, strides=poolsize, padding='valid', activation='relu', kernel_constraint=wsconstraint)(layer_conv[layer_number])))])


    #center of U
    if isinstance(kernel_size, (list,)):
        kernelsize = kernel_size[number_of_layers_half - 1]
    else:
        kernelsize = kernel_size

    if gn_type=='channels':
        groups = int(np.clip(np.round((expansion_rate**(number_of_layers_half-1))*starting_filter_number)/gn_param,1,np.round((expansion_rate**(number_of_layers_half-1))*starting_filter_number)))
    else:
        groups = int(np.clip(gn_param,1,np.round((expansion_rate**(number_of_layers_half-1))*starting_filter_number)))

    layer_conv[number_of_layers_half] = DRPT(rate=dropout_rate)(GroupNormalization(groups=groups)(Conv(filters=np.round((expansion_rate**(number_of_layers_half-1))*starting_filter_number), kernel_size=kernelsize, padding='same', activation='relu', kernel_constraint=wsconstraint)(layer_nonconv[number_of_layers_half-1])))
    for _ in range(1, (num_repeat_bottom_conv + 1)*num_conv_per_pool):
        layer_conv[number_of_layers_half] = DRPT(rate=dropout_rate)(GroupNormalization(groups=groups)(Conv(filters=np.round((expansion_rate**(number_of_layers_half-1))*starting_filter_number), kernel_size=kernelsize, padding='same', activation='relu', kernel_constraint=wsconstraint)(layer_conv[number_of_layers_half])))

    print('{:<30}'.format(str(layer_conv[number_of_layers_half].shape)),'\tgroups:', groups, '\tkernel:', kernelsize,'\tdroprate:', dropout_rate)

    #second half of U
    for layer_number in range(number_of_layers_half+1,2*number_of_layers_half):
        number_of_filters_current = np.round((expansion_rate**(2*number_of_layers_half-layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate * (np.power((number_of_filters_current / number_of_filters_max),dropout_power))

        if isinstance(pool_size,(list,)):
            poolsize=pool_size[2*number_of_layers_half-layer_number-1]
        else:
            poolsize=pool_size

        if isinstance(kernel_size, (list,)):
            kernelsize = kernel_size[2*number_of_layers_half-layer_number-1]
        else:
            kernelsize = kernel_size

        if gn_type=='channels':
            groups = int(np.clip(number_of_filters_current/gn_param,1,number_of_filters_current))
        else:
            groups = int(np.clip(gn_param,1,number_of_filters_current))

        layer_nonconv[layer_number]=Concatenate(axis=-1)([DRPT(rate=drop_rate_layer)(GroupNormalization(groups=groups)(Conv(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu', kernel_constraint=wsconstraint)(UpSampling(size=poolsize)(layer_conv[layer_number-1])))), layer_conv[2*number_of_layers_half-layer_number]])
        layer_conv[layer_number] = DRPT(rate=drop_rate_layer)(GroupNormalization(groups=groups)(Conv(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu', kernel_constraint=wsconstraint)(layer_nonconv[layer_number])))
        for _ in range(1, num_conv_per_pool):
            layer_conv[layer_number] = DRPT(rate=drop_rate_layer)(GroupNormalization(groups=groups)(Conv(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu', kernel_constraint=wsconstraint)(layer_conv[layer_number])))
        print('{:<30}'.format(str(layer_conv[layer_number].shape)),'\tgroups:', groups, '\tkernel:', kernelsize,'\tdroprate:',drop_rate_layer)

    #Add CNN with output
    if add_conv_layers > 0:
        print()
        print('Adding '+str(add_conv_layers)+' CNN layers to U-net')
        number_of_filters_current = add_conv_filter_number

        if isinstance(kernel_size, (list,)):
            kernelsize = kernel_size[0]
        else:
            kernelsize = kernel_size

        if gn_type=='channels':
            groups = int(np.clip(number_of_filters_current/gn_param,1,number_of_filters_current))
        else:
            groups = int(np.clip(gn_param,1,number_of_filters_current))

        if add_conv_dropout_rate is None:
            drop_rate_layer = dropout_rate * (np.power((number_of_filters_current / number_of_filters_max),dropout_power))
        else:
            drop_rate_layer = add_conv_dropout_rate

        for layer_CNN_number in range(add_conv_layers):
            layer_conv[2 * number_of_layers_half + layer_CNN_number] = DRPT(rate=drop_rate_layer)(GroupNormalization(groups=groups)(Conv(number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu', kernel_constraint=wsconstraint)(layer_conv[2 * number_of_layers_half + layer_CNN_number - 1])))

        print('{:<30}'.format(str(layer_conv[2 * number_of_layers_half + layer_CNN_number].shape)),'\tgroups:', groups, '\tkernel:', kernelsize, '\tdroprate:', drop_rate_layer)

    if isinstance(kernel_size, (list,)):
        kernelsize = kernel_size[0]
    else:
        kernelsize = kernel_size

    layer_conv[2 * number_of_layers_half + add_conv_layers] = Conv(channels_out, kernel_size=kernelsize, padding='same', activation=final_activation)(layer_conv[2 * number_of_layers_half + add_conv_layers - 1])
    print()
    print('Output:',layer_conv[2 * number_of_layers_half + add_conv_layers].shape)
    print()
    #build and compile U
    model = Model(inputs=[layer_nonconv[0]], outputs=[layer_conv[2 * number_of_layers_half + add_conv_layers]],name='unet')
    print('Successfully built ' + str(len(data_shape)) + 'D U-net model')
    return model