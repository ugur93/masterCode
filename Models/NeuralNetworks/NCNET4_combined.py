
from .base import *
from .base_class import NN_BASE



def sub(inputs):
    s = inputs[0]
    for i in range(1, len(inputs)):
        s -= inputs[i]
    return s

class NET4_W_PRESSURE(NN_BASE):


    def __init__(self,pressure_weights_path):

        self.model_name='NCNET3_PBH_INPUT'

        self.output_layer_activation = 'relu'
        # Training config
        optimizer = 'adam'  # SGD(momentum=0.9,nesterov=True)
        loss = huber
        self.activation='relu'
        nb_epoch = 1000
        batch_size = 64
        verbose = 0


        n_depth = 2
        n_width = 90
        l2w = 0.0003

        self.n_depth_pdc=2
        self.n_depth_pwh=2
        self.n_depth_pbh=1

        self.n_width_pdc=60
        self.n_width_pwh=90
        self.n_width_pbh=80
        self.add_thresholded_output=True




        self.input_tags = {}

        self.chk_names=['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
        self.well_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']  #
        tags = ['CHK']#, 'PBH', 'PWH', 'PDC']
        for name in self.well_names:
            if name not in ['C2','D1']:
                self.input_tags[name+'_PBH_IN']=[name+'_PBH']
            self.input_tags[name+'_CHK'] = []
            for tag in tags:
                if (name == 'C2' or name == 'D1') and tag == 'PBH':
                    pass
                else:
                    self.input_tags[name+'_CHK'].append(name + '_' + tag)

        self.input_tags['PRESSURE_INPUT']=[]
        for key in ['C1', 'C2', 'C3', 'C4', 'B3', 'B1', 'D1']:
            for tag in tags:
                self.input_tags['PRESSURE_INPUT'].append(key + '_' + tag)
        for key in ['C1','C3', 'C4','B1','B3']:
                self.input_tags['PRESSURE_INPUT'].append(key + '_' + '1_PBH')
        self.input_tags['RISER_B_CHK_INPUT']=['GJOA_RISER_OIL_B_CHK']



        self.output_tags = OIL_WELLS_QGAS_OUTPUT_TAGS

        for key in self.well_names:
            self.output_tags[key+'_PWH_out']=[key+'_PWH']
            self.output_tags[key + '_PDC_out'] = [key + '_PDC']


        self.loss_weights = {

            'C1_out': 0.0,
            'C2_out': 0.0,
            'C3_out': 0.0,
            'C4_out': 0.0,
            'D1_out': 0.0,
            'B3_out': 0.0,
            'B1_out': 0.0,

            'C1_PWH_out': 0.0,
            'C2_PWH_out': 0.0,
            'C3_PWH_out': 0.0,
            'C4_PWH_out': 0.0,
            'D1_PWH_out': 0.0,
            'B3_PWH_out': 0.0,
            'B1_PWH_out': 0.0,

            'C1_PDC_out': 0.0,
            'C2_PDC_out': 0.0,
            'C3_PDC_out': 0.0,
            'C4_PDC_out': 0.0,
            'D1_PDC_out': 0.0,
            'B3_PDC_out': 0.0,
            'B1_PDC_out': 0.0,




            'GJOA_TOTAL': 1.0
        }


        print(self.output_tags)
        super().__init__(n_width=n_width, n_depth=n_depth, l2_weight=l2w, dp_rate=0, seed=3014,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))


        rate_outputs = []

        pres_inputs,pres_outputs,pres_output_layers=self.generate_pressure_model()

        all_inputs=pres_inputs
        all_outputs=pres_outputs
        for key in self.well_names:
            n_input = len(self.input_tags[key+'_CHK'])
            aux_input, input,merged_output,aux_input_PBH = self.generate_input_module(n_input=n_input,pres_output_layers=pres_output_layers[key],name=key)



            all_inputs.append(input)
            all_inputs.append(aux_input)
            if aux_input_PBH is not None:
                all_inputs.append(aux_input_PBH)
            rate_outputs.append(merged_output)
            all_outputs.append(merged_output)


        merged_input = Add( name='GJOA_TOTAL')(rate_outputs)

        all_outputs=all_outputs+[merged_input]

        self.model = Model(input=all_inputs, output=all_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)

        #.model.load_weights(self.pressure_weights_path,by_name=True)


    def generate_input_module(self,n_input,pres_output_layers,name):
        chk_input = Input(shape=(n_input,), dtype='float32', name=name+'_CHK')

        PWH_out=pres_output_layers[0]
        PDC_out=pres_output_layers[1]
        aux_input_PBH=None
        if name in ['C2', 'D1']:
            all_inputs = Concatenate(name=name)([chk_input, PWH_out, PDC_out])
        else:

            aux_input_PBH = Input(shape=(1,), dtype='float32', name=name + '_PBH_IN')
            all_inputs = Concatenate(name=name)([chk_input, PWH_out, PDC_out,aux_input_PBH])
        #pres_output_layers.append(sub_input)



        output_layer = Dense(self.n_width, activation='relu', kernel_regularizer=l2(self.l2weight),
                            kernel_initializer=self.init, use_bias=True, name=name + '_0')(all_inputs)
        # temp_output = MaxoutDense(self.n_width,kernel_regularizer=l2(self.l2weight),kernel_initializer=self.init,use_bias=True)(input_layer)
        # temp_output=PReLU()(temp_output)
        for i in range(1, self.n_depth):
            if self.dp_rate > 0:
                output_layer = Dropout(self.dp_rate, name=name + '_dp_' + str(i))(output_layer)
            output_layer = Dense(self.n_width, activation='relu', kernel_regularizer=l2(self.l2weight),
                                kernel_initializer=self.init, use_bias=True, name=name + '_' + str(i))(output_layer)
            # temp_output = PReLU()(temp_output)

        output_layer = Dense(1, kernel_initializer=self.init, kernel_regularizer=l2(self.l2weight),
                             activation=self.output_layer_activation, use_bias=True, name=name + '_o')(output_layer)


        #aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        aux_input = Input(shape=(1,), dtype='float32', name='OnOff_' + name)
        merged_output=Multiply(name=name+'_out')([aux_input, output_layer])



        return aux_input, chk_input,merged_output,aux_input_PBH

    def update_model(self):
        self.nb_epoch = 10000
        self.output_layer_activation = 'relu'
        self.aux_inputs = []
        self.inputs = []
        self.merged_outputs = []
        self.outputs = []

        old_model = self.model
        self.initialize_model()
        weights = old_model.get_weights()
        self.model.set_weights(weights)

    def generate_pressure_model(self):

        all_chk_input = Input(shape=(len(self.input_tags['PRESSURE_INPUT']),), dtype='float32', name='PRESSURE_INPUT')
        riser_chk_input = Input(shape=(len(self.input_tags['RISER_B_CHK_INPUT']),), dtype='float32',
                                name='RISER_B_CHK_INPUT')

        all_and_riser_chk_input = Concatenate( name='RISER_MERGE')([all_chk_input, riser_chk_input])

        output_layers = {}
        outputs = []
        inputs = [all_chk_input, riser_chk_input]

        # sub_model_all = Dense(20, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all')(all_chk_input)
        # sub_model_all = Dense(20, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all2')(sub_model_all)


        for key in self.well_names:
            sub_model_PWH = self.generate_sub_pressure_model(all_chk_input, name=key + '_PWH',depth=self.n_depth_pwh,width=self.n_width_pwh)
            sub_model_PDC = self.generate_sub_pressure_model(all_and_riser_chk_input, name=key + '_PDC',depth=self.n_depth_pdc,width=self.n_width_pdc)
            # sub_model_temp = Dense(20, W_regularizer=l2(self.l2weight), activation='relu')(sub_model_temp)


            # sub_model_temp = Dropout(0.05)(sub_model1)
            PWH_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PWH_out',trainable=False)(sub_model_PWH)

            PDC_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PDC_out',trainable=False)(sub_model_PDC)



            # sub_model_out = Dense(1, W_regularizer=l2(self.l2weight), activation=self.out_act)(sub_model_out)

            output_layers[key] = [PWH_out,PDC_out]
            outputs.append(PWH_out)
            outputs.append(PDC_out)





        return inputs,outputs,output_layers

    def generate_sub_pressure_model(self,input_layer,name,depth=2,width=50):
        i=0
        sub_model = Dense(width, kernel_regularizer=l2(self.l2weight), activation='relu',name=name+'_'+str(i),trainable=False)(input_layer)
        for i in range(1,depth):
            # sub_model_temp=Dropout(0.01)(sub_model_temp)
            sub_model = Dense(width, kernel_regularizer=l2(self.l2weight), activation='relu',name=name+'_'+str(i),trainable=False)(sub_model)

        return sub_model


class NET4_W_PRESSURE2(NN_BASE):


    def __init__(self,pressure_weights_path):

        self.model_name='NCNET3_PBH_INPUT'

        self.output_layer_activation = 'relu'
        # Training config
        optimizer = 'adam'  # SGD(momentum=0.9,nesterov=True)
        loss = 'mae'
        self.activation='relu'
        nb_epoch = 1000
        batch_size = 64
        verbose = 0


        n_depth = 2
        n_width = 100
        l2w = 0.00017

        self.n_depth_pdc=2
        self.n_depth_pwh=3
        self.n_depth_pbh=3

        self.n_width_pdc=50
        self.n_width_pwh=50
        self.n_width_pbh=50
        self.add_thresholded_output=True




        self.input_tags = {}
        self.well_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']  #
        tags = ['CHK']#, 'PBH', 'PWH', 'PDC']
        for name in self.well_names:
            if name not in ['C2','D1']:
                self.input_tags[name+'_PBH_IN']=[name+'_PBH']
            self.input_tags[name+'_CHK'] = []
            for tag in tags:
                if (name == 'C2' or name == 'D1') and tag == 'PBH':
                    pass
                else:
                    self.input_tags[name+'_CHK'].append(name + '_' + tag)

        self.input_tags['PRESSURE_INPUT']=[]
        for key in self.well_names:
            for tag in tags:
                self.input_tags['PRESSURE_INPUT'].append(key + '_' + tag)
        #for key in ['C1','C3', 'C4','B1','B3']:
        #        self.input_tags['PRESSURE_INPUT'].append(key + '_' + '1_PBH')
        self.input_tags['RISER_B_CHK_INPUT']=['GJOA_RISER_OIL_B_CHK']



        self.output_tags = OIL_WELLS_QGAS_OUTPUT_TAGS

        for key in self.well_names:
            self.output_tags[key+'_PWH_out']=[key+'_PWH']
            self.output_tags[key + '_PDC_out'] = [key + '_PDC']
            if key not in ['D1','C2']:
                self.output_tags[key + '_PBH_out'] = [key + '_PBH']


        self.loss_weights = {

            'C1_out': 0.0,
            'C2_out': 0.0,
            'C3_out': 0.0,
            'C4_out': 0.0,
            'D1_out': 0.0,
            'B3_out': 0.0,
            'B1_out': 0.0,

            'C1_PWH_out': 0.0,
            'C2_PWH_out': 0.0,
            'C3_PWH_out': 0.0,
            'C4_PWH_out': 0.0,
            'D1_PWH_out': 0.0,
            'B3_PWH_out': 0.0,
            'B1_PWH_out': 0.0,

            'C1_PDC_out': 0.0,
            'C2_PDC_out': 0.0,
            'C3_PDC_out': 0.0,
            'C4_PDC_out': 0.0,
            'D1_PDC_out': 0.0,
            'B3_PDC_out': 0.0,
            'B1_PDC_out': 0.0,

            'C1_PBH_out': 0.0,
            #'C2_PBH_out': 0.0,
            'C3_PBH_out': 0.0,
            'C4_PBH_out': 0.0,
            #'D1_PBH_out': 0.0,
            'B3_PBH_out': 0.0,
            'B1_PBH_out': 0.0,




            'GJOA_TOTAL': 1.0
        }


        print(self.output_tags)
        super().__init__(n_width=n_width, n_depth=n_depth, l2_weight=l2w, dp_rate=0, seed=3014,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))


        rate_outputs = []

        pres_inputs,pres_outputs,pres_output_layers=self.generate_pressure_model()

        all_inputs=pres_inputs
        all_outputs=pres_outputs
        for key in self.well_names:
            n_input = len(self.input_tags[key+'_CHK'])
            aux_input, input,merged_output,aux_input_PBH = self.generate_input_module(n_input=n_input,pres_output_layers=pres_output_layers[key],name=key)



            all_inputs.append(input)
            all_inputs.append(aux_input)
            if aux_input_PBH is not None:
                all_inputs.append(aux_input_PBH)
            rate_outputs.append(merged_output)
            all_outputs.append(merged_output)


        merged_input = Add( name='GJOA_TOTAL')(rate_outputs)

        all_outputs=all_outputs+[merged_input]

        self.model = Model(input=all_inputs, output=all_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)

        #.model.load_weights(self.pressure_weights_path,by_name=True)


    def generate_input_module(self,n_input,pres_output_layers,name):
        chk_input = Input(shape=(n_input,), dtype='float32', name=name+'_CHK')

        PWH_out=pres_output_layers[0]
        PDC_out=pres_output_layers[1]
        aux_input_PBH=None
        if name in ['C2', 'D1']:
            all_inputs = Concatenate(name=name)([chk_input, PWH_out, PDC_out])
        else:
            PBH_out = pres_output_layers[2]

            all_inputs = Concatenate(name=name)([chk_input,PBH_out, PWH_out,PDC_out])
        #pres_output_layers.append(sub_input)



        output_layer = Dense(self.n_width, activation='relu', kernel_regularizer=l2(self.l2weight),
                            kernel_initializer=self.init, use_bias=True, name=name + '_0')(all_inputs)
        # temp_output = MaxoutDense(self.n_width,kernel_regularizer=l2(self.l2weight),kernel_initializer=self.init,use_bias=True)(input_layer)
        # temp_output=PReLU()(temp_output)
        for i in range(1, self.n_depth):
            if self.dp_rate > 0:
                output_layer = Dropout(self.dp_rate, name=name + '_dp_' + str(i))(output_layer)
            output_layer = Dense(self.n_width, activation='relu', kernel_regularizer=l2(self.l2weight),
                                kernel_initializer=self.init, use_bias=True, name=name + '_' + str(i))(output_layer)
            # temp_output = PReLU()(temp_output)

        output_layer = Dense(1, kernel_initializer=self.init, kernel_regularizer=l2(self.l2weight),
                             activation=self.output_layer_activation, use_bias=True, name=name + '_o')(output_layer)


        #aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        aux_input = Input(shape=(1,), dtype='float32', name='OnOff_' + name)
        merged_output=Multiply(name=name+'_out')([aux_input, output_layer])



        return aux_input, chk_input,merged_output,aux_input_PBH

    def update_model(self):
        self.nb_epoch = 10000
        self.output_layer_activation = 'relu'
        self.aux_inputs = []
        self.inputs = []
        self.merged_outputs = []
        self.outputs = []

        old_model = self.model
        self.initialize_model()
        weights = old_model.get_weights()
        self.model.set_weights(weights)

    def generate_pressure_model(self):

        all_chk_input = Input(shape=(len(self.input_tags['PRESSURE_INPUT']),), dtype='float32', name='PRESSURE_INPUT')
        riser_chk_input = Input(shape=(len(self.input_tags['RISER_B_CHK_INPUT']),), dtype='float32',
                                name='RISER_B_CHK_INPUT')

        all_and_riser_chk_input = Concatenate( name='RISER_MERGE')([all_chk_input, riser_chk_input])

        output_layers = {}
        outputs = []
        inputs = [all_chk_input, riser_chk_input]

        # sub_model_all = Dense(20, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all')(all_chk_input)
        # sub_model_all = Dense(20, W_regularizer=l2(self.l2weight), activation='relu',name='pres_sub_all2')(sub_model_all)


        for key in self.well_names:
            sub_model_PWH = self.generate_sub_pressure_model(all_chk_input, name=key + '_PWH',depth=self.n_depth_pwh,width=self.n_width_pwh)
            sub_model_PDC = self.generate_sub_pressure_model(all_and_riser_chk_input, name=key + '_PDC',depth=self.n_depth_pdc,width=self.n_width_pdc)
            sub_model_PBH = self.generate_sub_pressure_model(all_chk_input, name=key + '_PBH',depth=self.n_depth_pbh,width=self.n_width_pbh)

            # sub_model_temp = Dense(20, W_regularizer=l2(self.l2weight), activation='relu')(sub_model_temp)


            # sub_model_temp = Dropout(0.05)(sub_model1)
            PWH_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PWH_out',trainable=False)(sub_model_PWH)

            PDC_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PDC_out',trainable=False)(sub_model_PDC)
            if key not in ['C2','D1']:
                PBH_out = Dense(1,
                            kernel_regularizer=l2(self.l2weight), activation='linear', name=key + '_PBH_out',trainable=False)(sub_model_PBH)
            # sub_model_out = Dense(1, W_regularizer=l2(self.l2weight), activation=self.out_act)(sub_model_out)
                outputs.append(PBH_out)
                output_layers[key] = [PWH_out, PDC_out, PBH_out]
            else:
                output_layers[key] = [PWH_out, PDC_out]
            outputs.append(PWH_out)
            outputs.append(PDC_out)






        return inputs,outputs,output_layers

    def generate_sub_pressure_model(self,input_layer,name,depth=2,width=50):
        i=0
        sub_model = Dense(width, kernel_regularizer=l2(self.l2weight), activation='relu',name=name+'_'+str(i),trainable=False)(input_layer)
        for i in range(1,depth):
            # sub_model_temp=Dropout(0.01)(sub_model_temp)
            sub_model = Dense(width, kernel_regularizer=l2(self.l2weight), activation='relu',name=name+'_'+str(i),trainable=False)(sub_model)

        return sub_model

class NET4_W_PRESSURE3(NN_BASE):

    def __init__(self):

        self.model_name = 'NCNET3_PBH_INPUT'

        self.output_layer_activation = 'relu'
        # Training config
        optimizer = 'adam'  # SGD(momentum=0.9,nesterov=True)
        loss = huber
        self.activation = 'relu'
        nb_epoch = 1000
        batch_size = 64
        verbose = 0

        n_depth = 2
        n_width = 90
        l2w = 0.00015

        self.n_depth_pdc = 2
        self.n_depth_pwh = 2
        self.n_depth_pbh = 1

        self.n_width_pdc = 60
        self.n_width_pwh = 90
        self.n_width_pbh = 80
        self.presl2weight=0
        self.add_thresholded_output = True

        self.input_tags = {}
        self.well_names = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']  #
        tags = ['CHK']  # , 'PBH', 'PWH', 'PDC']
        for name in self.well_names:
            if name not in ['C2', 'D1']:
                self.input_tags[name + '_PBH_IN'] = [name + '_PBH']
            self.input_tags[name + '_CHK'] = []
            for tag in tags:
                if (name == 'C2' or name == 'D1') and tag == 'PBH':
                    pass
                else:
                    self.input_tags[name + '_CHK'].append(name + '_' + tag)

        self.input_tags['CHK_INPUT_NOW'] = []
        self.input_tags['CHK_INPUT_PREV'] = []
        #if tag == 'PDC':
        self.input_tags['CHK_INPUT_NOW']= ['GJOA_RISER_OIL_B_CHK']
        self.input_tags['CHK_INPUT_PREV']=['GJOA_RISER_OIL_B_shifted_CHK']
        for key in self.well_names:
            for tag in tags:
                self.input_tags['CHK_INPUT_NOW'].append(key + '_' + tag)
                self.input_tags['CHK_INPUT_PREV'].append(key + '_shifted_' + tag)

                self.input_tags['SHIFTED_PRESSURE_PWH_' + key] = [key + '_shifted_PWH']
                self.input_tags['SHIFTED_PRESSURE_PDC_' + key] = [key + '_shifted_PDC']
                if key not in ['C2', 'D1']:
                    self.input_tags['SHIFTED_PRESSURE_PBH_' + key] = [key + '_shifted_PBH']
        # for key in ['C1','C3', 'C4','B1','B3']:
        #        self.input_tags['PRESSURE_INPUT'].append(key + '_' + '1_PBH')
        #self.input_tags['CHK_INPUT'].append('GJOA_RISER_OIL_B_CHK')

        self.output_tags = OIL_WELLS_QGAS_OUTPUT_TAGS

        for key in self.well_names:
            #print(key)
            self.output_tags[key + '_PWH_out2'] = [key + '_PWH']
            self.output_tags[key + '_PDC_out2'] = [key + '_PDC']
            if key not in ['C2','D1']:
                self.output_tags[key + '_PBH_out2'] = [key + '_PBH']
        #print(self.output_tags)
        self.loss_weights = {

            'C1_out': 0.0,
            'C2_out': 0.0,
            'C3_out': 0.0,
            'C4_out': 0.0,
            'D1_out': 0.0,
            'B3_out': 0.0,
            'B1_out': 0.0,

            'C1_PBH_out2':0.0,
            #'C2_PBH_out2': 0.0,
            'C3_PBH_out2': 0.0,
            'C4_PBH_out2': 0.0,
            'B1_PBH_out2': 0.0,
            'B3_PBH_out2': 0.0,
            #'D1_PBH_out2': 0.0,

            'C1_PDC_out2': 0.0,
            'C2_PDC_out2': 0.0,
            'C3_PDC_out2': 0.0,
            'C4_PDC_out2': 0.0,
            'B1_PDC_out2': 0.0,
            'B3_PDC_out2': 0.0,
            'D1_PDC_out2': 0.0,

            'C1_PWH_out2': 0.0,
            'C2_PWH_out2': 0.0,
            'C3_PWH_out2': 0.0,
            'C4_PWH_out2': 0.0,
            'B1_PWH_out2': 0.0,
            'B3_PWH_out2': 0.0,
            'D1_PWH_out2': 0.0,


            'GJOA_TOTAL': 1.0
        }

        print(self.output_tags)
        super().__init__(n_width=n_width, n_depth=n_depth, l2_weight=l2w, dp_rate=0, seed=3014,
                         optimizer=optimizer, loss=loss, nb_epoch=nb_epoch, batch_size=batch_size)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        rate_outputs = []

        pres_inputs, pres_outputs, pres_output_layers = self.generate_pressure_model()

        all_inputs = pres_inputs
        all_outputs = pres_outputs
        for key in self.well_names:
            n_input = len(self.input_tags[key + '_CHK'])
            aux_input, input, merged_output = self.generate_input_module(n_input=n_input,
                                                                                        pres_output_layers=
                                                                                        pres_output_layers[key],
                                                                                        name=key)

            all_inputs.append(input)
            all_inputs.append(aux_input)

            rate_outputs.append(merged_output)
            all_outputs.append(merged_output)

        merged_input = Add(name='GJOA_TOTAL')(rate_outputs)

        all_outputs = all_outputs + [merged_input]

        self.model = Model(input=all_inputs, output=all_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
        print(self.model.summary())
        exit()
        # .model.load_weights(self.pressure_weights_path,by_name=True)

    def generate_input_module(self, n_input, pres_output_layers, name):
        chk_input = Input(shape=(n_input,), dtype='float32', name=name + '_CHK')


        PWH_out = pres_output_layers[0]
        PDC_out = pres_output_layers[1]

        if name in ['C1', 'C3', 'C4', 'B1', 'B3']:
            PBH_out = pres_output_layers[2]
            all_inputs = Concatenate(name=name)([chk_input, PBH_out,PWH_out,PDC_out])
        else:
            all_inputs = Concatenate(name=name)([chk_input, PWH_out,PDC_out])



        # pres_output_layers.append(sub_input)



        output_layer = Dense(self.n_width, activation='relu', kernel_regularizer=l2(self.l2weight),
                             kernel_initializer=self.init,trainable=True, use_bias=True, name=name + '_0')(all_inputs)
        # temp_output = MaxoutDense(self.n_width,kernel_regularizer=l2(self.l2weight),kernel_initializer=self.init,use_bias=True)(input_layer)
        # temp_output=PReLU()(temp_output)
        for i in range(1, self.n_depth):
            if self.dp_rate > 0:
                output_layer = Dropout(self.dp_rate, name=name + '_dp_' + str(i))(output_layer)
            output_layer = Dense(self.n_width, activation='relu',trainable=True, kernel_regularizer=l2(self.l2weight),
                                 kernel_initializer=self.init, use_bias=True, name=name + '_' + str(i))(
                output_layer)
            # temp_output = PReLU()(temp_output)

        output_layer = Dense(1, kernel_initializer=self.init,trainable=True, kernel_regularizer=l2(self.l2weight),
                             activation=self.output_layer_activation, use_bias=True, name=name + '_o')(output_layer)

        # aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        aux_input = Input(shape=(1,), dtype='float32', name='OnOff_' + name)
        merged_output = Multiply(name=name + '_out')([aux_input, output_layer])

        return aux_input, chk_input, merged_output

    def update_model(self):
        self.nb_epoch = 10000
        self.output_layer_activation = 'relu'
        self.aux_inputs = []
        self.inputs = []
        self.merged_outputs = []
        self.outputs = []

        old_model = self.model
        self.initialize_model()
        weights = old_model.get_weights()
        self.model.set_weights(weights)

    def generate_pressure_model(self):

        chk_input_now = Input(shape=(len(self.input_tags['CHK_INPUT_NOW']),), dtype='float32',
                              name='CHK_INPUT_NOW')
        chk_input_prev = Input(shape=(len(self.input_tags['CHK_INPUT_PREV']),), dtype='float32',
                               name='CHK_INPUT_PREV')

        #chk_input_now_riser = Input(shape=(len(self.input_tags['CHK_INPUT_NOW_RISER']),), dtype='float32',
        #                      name='CHK_INPUT_NOW_RISER')
        #chk_input_prev_riser = Input(shape=(len(self.input_tags['CHK_INPUT_PREV_RISER']),), dtype='float32',
        #                       name='CHK_INPUT_PREV_RISER')

        chk_delta = Add(name='CHK_DELTA')([chk_input_now, chk_input_prev])
        #chk_delta_riser = Add(name='CHK_DELTA_RISER')([chk_input_now_riser, chk_input_prev_riser])

        #chk_delta_riser=Concatenate(name='CHK_INPUT_RISER')([chk_delta_riser,chk_delta])
        output_layers = {}
        outputs = []
        inputs = [chk_input_now, chk_input_prev]

        for key in ['C1', 'C2', 'C3', 'C4', 'B3', 'B1', 'D1']:
            sub_model_PWH = generate_pressure_sub_model(chk_delta, name=key + '_PWH', depth=self.n_depth_pwh,
                                                        n_width=self.n_width_pwh, dp_rate=self.dp_rate, init=self.init,
                                                        l2weight=self.presl2weight)

            sub_model_PBH = generate_pressure_sub_model(chk_delta, name=key + '_PBH', depth=self.n_depth_pbh,
                                                        n_width=self.n_width_pbh, dp_rate=self.dp_rate, init=self.init,
                                                        l2weight=self.presl2weight)
            sub_model_PDC = generate_pressure_sub_model(chk_delta, name=key + '_PDC', depth=self.n_depth_pdc,
                                                        n_width=self.n_width_pdc, dp_rate=self.dp_rate, init=self.init,
                                                        l2weight=self.presl2weight)

            shifted_pressure_input_PWH = Input(shape=(len(self.input_tags['SHIFTED_PRESSURE_PWH_' + key]),), dtype='float32',
                                           name='SHIFTED_PRESSURE_PWH_' + key)
            shifted_pressure_input_PDC = Input(shape=(len(self.input_tags['SHIFTED_PRESSURE_PDC_' + key]),), dtype='float32',
                                          name='SHIFTED_PRESSURE_PDC_' + key)


            PWH_out = Dense(1,
                            kernel_regularizer=l2(self.presl2weight), activation='linear', name=key + '_PWH_out',
                            kernel_initializer=self.init)(sub_model_PWH)


            PDC_out = Dense(1,
                            kernel_regularizer=l2(self.presl2weight), activation='linear', name=key + '_PDC_out',
                            kernel_initializer=self.init)(sub_model_PDC)

            PWH_out = Add(name=key + '_PWH_out2')([PWH_out, shifted_pressure_input_PWH])

            PDC_out = Add(name=key + '_PDC_out2')([PDC_out, shifted_pressure_input_PDC])

            if key in ['C1', 'C3', 'C4', 'B1', 'B3']:
                shifted_pressure_input_PBH = Input(shape=(len(self.input_tags['SHIFTED_PRESSURE_PBH_' + key]),),
                                                  dtype='float32',
                                                  name='SHIFTED_PRESSURE_PBH_' + key)
                PBH_out = Dense(1,
                                kernel_regularizer=l2(self.presl2weight), activation='linear', name=key + '_PBH_out',
                                kernel_initializer=self.init)(sub_model_PBH)
                PBH_out = Add(name=key + '_PBH_out2')([PBH_out, shifted_pressure_input_PBH])

                outputs.append(PBH_out)
                output_layers[key] = [PWH_out, PDC_out,PBH_out]
                inputs.append(shifted_pressure_input_PBH)
            else:
                output_layers[key] = [PWH_out, PDC_out]

            outputs.append(PWH_out)
            outputs.append(PDC_out)

            inputs.append(shifted_pressure_input_PWH)
            inputs.append(shifted_pressure_input_PDC)





        return inputs, outputs, output_layers

    def generate_sub_pressure_model(self, input_layer, name, depth=2, width=50):
        i = 0
        sub_model = Dense(width, kernel_regularizer=l2(self.presl2weight), activation='relu', name=name + '_' + str(i),
                          trainable=True)(input_layer)
        for i in range(1, depth):
            # sub_model_temp=Dropout(0.01)(sub_model_temp)
            sub_model = Dense(width, kernel_regularizer=l2(self.l2weight), activation='relu',
                              name=name + '_' + str(i), trainable=True)(sub_model)

        return sub_model