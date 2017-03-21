
from .base import *
from .base_class import NN_BASE

import keras.backend as K
K.set_image_dim_ordering('th')
#GOOD

#incept 3
# dept 1
# width 20
#l2 0.0001
#opt rmsprop


#FOR OIL DATASET
#2x30 Dense
#Maxnorm 4 for all
#nbepoch=100
#batch=64


class NCNET1_GJOA2(NN_BASE):



    def __init__(self,n_depth=5,n_width=20,l2w=0.00001):

        self.model_name='NCNET2_OIL_QGAS_BEST_MODEL_2x20_l20c000005_incept4'


        self.output_layer_activation='linear'

        # Training config
        self.optimizer = 'adam'
        self.loss = 'mse'
        self.nb_epoch = 1
        self.batch_size = 64
        self.verbose = 0
        self.reg_constraint=False

        #Model config


        #Input module config
        self.n_inception =0
        self.n_depth = n_depth
        self.n_depth_incept=3
        self.n_width_incept=50
        self.n_width = n_width


        self.l2weight = l2w


        self.make_same_model_for_all=True
        self.add_thresholded_output=True

        self.input_tags = {}

        self.well_names = ['C1','C2', 'C3', 'C4','B1','B3','D1']

        tags = ['CHK','PWH','PBH','PDC']

        for name in self.well_names:

            self.input_tags[name] = []
            for tag in tags:
                if (name=='C2' or name=='D1') and tag=='PBH':
                    pass
                else:
                    self.input_tags[name].append(name + '_' + tag)


        OUT='GAS'
        if OUT=='GAS':
            self.output_tags = {


                'C1_out': ['C1_QGAS'],
                'C2_out': ['C2_QGAS'],
                'C3_out': ['C3_QGAS'],
                'C4_out': ['C4_QGAS'],
                'D1_out': ['D1_QGAS'],
                'B3_out': ['B3_QGAS'],
                'B1_out': ['B1_QGAS'],


                'GJOA_TOTAL':['GJOA_OIL_QGAS']
            }
        else:
            self.output_tags = {
                 'C1_out':['C1_QOIL'],
                 'C2_out':['C2_QOIL'],
                 'C3_out':['C3_QOIL'],
                 'C4_out':['C4_QOIL'],
                 'D1_out':['D1_QOIL'],
                 'B3_out':['B3_QOIL'],
                 'B1_out':['B1_QOIL'],
                 'GJOA_TOTAL': ['GJOA_TOTAL_SUM_QOIL']
            }
        self.loss_weights = {
            'B1_out':  0.0,
            'B3_out':  0.0,
            'C2_out':  0.0,
            'C3_out':  0.0,
            'D1_out':  0.0,
            'C4_out':  0.0,
            'C1_out':  0.0,

            'GJOA_TOTAL': 1.0,


        }


        super().__init__()

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        aux_inputs=[]
        inputs=[]
        merged_outputs=[]
        outputs=[]
        #merged_outputs=[]

        n_depth=self.n_depth
        n_width=self.n_width
        l2w=self.l2weight
        for key in self.well_names:
            n_input=len(self.input_tags[key])
            aux_input,input,merged_out,out=self.generate_input_module(n_depth=n_depth, n_width=n_width,
                                                                    n_input=n_input, n_inception=self.n_inception,
                                                                    l2_weight=l2w, name=key,thresholded_output=self.add_thresholded_output)
            aux_inputs.append(aux_input)
            inputs.append(input)
            merged_outputs.append(merged_out)
            outputs.append(out)


        merged_input = add(merged_outputs,name='GJOA_TOTAL')

        all_outputs=merged_outputs+[merged_input]
        all_inputs = inputs

        if self.add_thresholded_output:
            all_inputs+=aux_inputs
        print(all_inputs)

        self.model = Model(inputs=all_inputs, outputs=all_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)




    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):
        K.set_image_dim_ordering('th')
        input_layer = Input(shape=(n_input,), dtype='float32', name=name)

        temp_output = Dense(self.n_width,  activation='relu',kernel_regularizer=l2(self.l2weight),kernel_initializer=INIT,use_bias=True)(input_layer)

        for i in range(1,self.n_depth):
            temp_output = Dense(self.n_width, activation='relu',kernel_regularizer=l2(self.l2weight),kernel_initializer=INIT,use_bias=True)(temp_output)


        output_layer = Dense(1, kernel_initializer=INIT,kernel_regularizer=l2(self.l2weight),activation=self.output_layer_activation, use_bias=True)(temp_output)

        aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)

        return aux_input, input_layer, merged_output, output_layer

    def generate_input_module_gas(self, n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):
        K.set_image_dim_ordering('th')
        input_layer = Input(shape=(n_input,), dtype='float32', name=name)

        mod=Dense(50, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
                     use_bias=True,trainable=True)(input_layer)
        mod1 = Dense(10, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
                     use_bias=True,trainable=True)(mod)
        for i in range(1, self.n_depth):
            mod1 = Dense(30, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
                         use_bias=True,trainable=True)(mod1)
        #mod1 = Dense(1, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
        #            use_bias=True, trainable=True)(mod1)



        mod2 = Dense(10, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
                     use_bias=True, trainable=True)(mod)

        for i in range(1, self.n_depth):
            mod2 = Dense(20, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
                              use_bias=True, trainable=True)(mod2)
        mod2 = Dense(20, activation='relu', kernel_regularizer=l2(self.l2weight), kernel_initializer=INIT,
                     use_bias=True, trainable=True)(mod2)

        main_model = concatenate([mod1, mod2])



        output_layer = Dense(1, kernel_initializer=INIT, kernel_regularizer=l2(self.l2weight), activation=self.output_layer_activation,
                             use_bias=True)(main_model)

        aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)


        return aux_input, input_layer, merged_output, output_layer

    def generate_input_module_oil(self, n_depth, n_width, l2_weight, name, n_input, thresholded_output,
                                  n_inception=0):
        K.set_image_dim_ordering('th')
        input_layer = Input(shape=(n_input,), dtype='float32', name=name)

        #mod_dense = Flatten()(input_layer)
        mod_dense = Dense(self.n_width, activation='relu', W_constraint=maxnorm(self.maxnorm[0]), init=INIT,
                          bias=True)(input_layer)
        for i in range(1, self.n_depth):
            mod_dense = Dense(self.n_width, activation='relu', W_constraint=maxnorm(self.maxnorm[0]), init=INIT,
                              bias=True)(mod_dense)


        #mod_conv = Dropout(0.5)(input_layer)

        mod_conv = Dense(20, activation='relu', init=INIT,
                         bias=True, W_constraint=maxnorm(self.maxnorm[0]))(input_layer)
        mod_conv = Dropout(0.5)(mod_conv)

        mod_conv = Dense(20, activation='relu', init=INIT,
                         bias=True, W_constraint=maxnorm(self.maxnorm[0]))(mod_conv)

        #mod_conv = Dropout(0.5)(mod_conv)

        #mod_conv = Dense(30, activation='relu', init=INIT,
        #                 bias=True, W_constraint=maxnorm(5))(mod_conv)

        #mod_conv = Dropout(0.1)(mod_conv)



        #mod_conv = Flatten()(mod_conv)
        main_model = merge([mod_conv, mod_dense], mode='concat')



        output_layer = Dense(1, init=INIT, W_constraint=maxnorm(self.maxnorm[0]),
                             activation=self.output_layer_activation,
                             bias=True)(main_model)

        aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)

        return aux_input, input_layer, merged_output, output_layer

    def update_model(self):
        self.nb_epoch=10000
        self.output_layer_activation='relu'
        self.aux_inputs=[]
        self.inputs=[]
        self.merged_outputs=[]
        self.outputs=[]

        old_model=self.model
        self.initialize_model()
        weights=old_model.get_weights()
        self.model.set_weights(weights)


