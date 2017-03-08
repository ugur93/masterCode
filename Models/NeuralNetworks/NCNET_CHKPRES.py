
from .base import *
from .base_class import NN_BASE

import keras.backend as K

def sub(inputs):
    s = inputs[0]
    for i in range(1, len(inputs)):
        s -= inputs[i]
    return s
PRES='PWH'
from theano.ifelse import ifelse
from theano import tensor as T
import theano
def zero_layer(x):
    a,b=T.scalars('a','b')
    c, y = T.matrices('v', 'y')

    z_lazy=ifelse(T.lt(a,b),c,y)

    f_lazyifelse= theano.function([a,b,c,y], z_lazy,
                               mode=theano.Mode(linker='vm'),allow_input_downcast=True)
    return f_lazyifelse(5,x,0,x)
    #+return x
class SSNET3_PRESSURE(NN_BASE):


    def __init__(self,Data,mno,mnh,depth):



        self.model_name='GJOA_OIL_DO2_'+PRES
        self.out_act='linear'
        # Training config
        self.optimizer ='adam'#SGD(momentum=0.9,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 10000
        self.batch_size = 64
        self.verbose = 0
        self.mno=mno
        self.mnh=mnh
        self.p_dropout=1

        self.n_inputs=5
        self.n_outputs=1
        self.SCALE=100000
        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = depth
        self.n_width = 1
        self.l2weight = 0.0000
        self.add_thresholded_output=False



        self.input_name='E1'
        self.well_names=['F1','B2','D3','E1']
        chk_names=['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
        if PRES=='PBH':
            self.well_names=['C1','C3', 'C4','B1','B3']
        else:
            self.well_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
        tags=['CHK']

        self.input_tags={'Main_input':[]}
        for key in chk_names:
            for tag in tags:
                self.input_tags['Main_input'].append(key+'_'+tag)
        #self.input_tags['Main_input'].append('GJOA_RISER_OIL_B_CHK')
        self.n_inputs = len(self.input_tags['Main_input'])
        self.n_outputs=1

        self.output_tags = {}
        for name in self.well_names:
            self.output_tags[name + '_out'] = [name + '_'+PRES]
            #self.input_tags['aux_' + name] = [name + '_CHK_zero']
            self.input_tags['aux_' + name] = [name + '_CHK']
        self.initialize_zero_thresholds(Data)
        super().__init__()
    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        chk_input = Input(shape=(1,len(self.input_tags['Main_input'])), dtype='float32', name='Main_input')
        #
        inputs = [chk_input]
        outputs = []

        for key in self.well_names:


                sub_model = Flatten()(chk_input)

                sub_model = Dense(100, W_constraint=maxnorm(1), activation='relu')(sub_model)


                sub_model = Dense(1, W_constraint=maxnorm(1),activation=self.out_act)(sub_model)

                aux_input = Input(shape=(len(self.input_tags['aux_' + key]),), dtype='float32',name='aux_' + key)
                #aux_input_act = ThresholdedReLU(theta=self.chk_thresholds)(aux_input)
                sub_model_out = merge([sub_model, aux_input], mode='mul', name=key + '_out')
                #sub_model=Activation(zero_layer, name=key + '_out')(sub_model)
                outputs.append(sub_model_out)
                inputs.append(aux_input)

        self.model = Model(input=inputs, output=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def update_model(self):
        self.nb_epoch=10000
        self.out_act='relu'
        #self.aux_inputs=[]
        #self.inputs=[]
        #self.merged_outputs=[]
        #self.outputs=[]

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)