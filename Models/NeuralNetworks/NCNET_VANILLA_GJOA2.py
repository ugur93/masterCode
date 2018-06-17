
from .base import *
from .base_class import NN_BASE



#GOOD

#incept 3
# dept 1
# width 20
#l2 0.0001
#opt rmsprop
class NCNET_VANILLA(NN_BASE):


    def __init__(self):

        self.model_name='NCNET1_VANILLA_GJOA2'


        self.n_inputs=7
        #self.n_outputs=7
        self.SCALE=100

        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = 2
        self.n_width = 20
        self.l2weight = 0.0001
        self.add_thresholded_output=False

        self.input_tags = {}
        input_name='MAIN_INPUT'
        self.well_names = ['C1','C2', 'C3', 'C4', 'B3', 'B1', 'D1']
        tags = ['CHK','PDC','PWH', 'PBH']
        self.input_tags[input_name] = []
        for name in self.well_names:
            for tag in tags:
                if (name == 'C2' or name=='D1') and tag == 'PBH':
                    pass
                else:
                    self.input_tags[input_name].append(name + '_' + tag)
        #self.input_tags[input_name].append('C1_PDC')
        #self.input_tags[input_name].append('time')
        print(self.input_tags)
        self.n_inputs=len(self.input_tags[input_name])


        # self.input_tags[name].append('GJOA_RISER_OIL_B_CHK')

        # self.input_tags['D1']=['D1_CHK','D1_PDC','D1_PWH']
        # self.input_tags['C2'] = ['C2_CHK', 'C2_PDC', 'C2_PWH']
        # self.input_tags['Riser']=['GJOA_RISER_OIL_B_CHK','GJOA_RISER_OIL_B_PDC']

        cols_out=[]
        out_tags=['PWH']#,'PWH']
        for name in self.well_names:
            for tag in out_tags:
                cols_out.append(name+'_'+tag)
        self.output_tags = {
            'C1_out': ['C1_QOIL'],
            'C2_out': ['C2_QOIL'],
            'C3_out': ['C3_QOIL'],
            'C4_out': ['C4_QOIL'],
            'D1_out': ['D1_QOIL'],
            'B3_out': ['B3_QOIL'],
            'B1_out': ['B1_QOIL'],
            # 'Riser_out':['GJOA_TOTAL_QOIL_SUM'],
            #'TOTAL_OUT': ['C1_QOIL','C2_QOIL','C3_QOIL','C4_QOIL','B1_QOIL','B3_QOIL','D1_QOIL','GJOA_TOTAL_QOIL']
            'TOTAL_OUT':['GJOA_OIL_QGAS']
            #'TOTAL_OUT':cols_out
        }


        self.loss_weights=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]

        #Training config
        self.optimizer = 'adam' #SGD(momentum=0.9,nesterov=True)
        self.loss = 'mse'
        self.nb_epoch = 10000
        self.batch_size = 64
        self.verbose = 0




        super().__init__()

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))

        main_input = Input(shape=(self.n_inputs,), dtype='float32', name='MAIN_INPUT')

        #main_model=add_layers(main_input,self.n_depth,self.n_width,self.l2weight)
        print(self.n_outputs)

        main_output = []
        for key in self.output_tags.keys():
            main_model = add_layers(main_input, n_depth=self.n_depth, n_width=self.n_width, l2_weight=self.l2weight)
            # main_output=merge([PWH_out,PDC_out],mode='sum',name='MAIN_OUTPUT')
            main_output.append(Dense(len(self.output_tags[key]),  name=key)(main_model))
        #out=Dense(self.n_outputs,name='TOTAL_OUT')(main_model)

        self.model = Model(input=main_input, output=main_output)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):

        input_layer = Input(shape=(n_input,), dtype='float32', name=name)
        # temp_output=Dropout(0.1)(input_layer)

        if n_depth == 0:
            temp_output = input_layer
        else:
            if n_inception > 1:
                #temp_output = add_layers(input_layer, 1, n_width, l2_weight)
                temp_output = generate_inception_module(input_layer, n_inception, n_depth, n_width, l2_weight)
                temp_output = add_layers(temp_output, 2, n_width, l2_weight)
                #temp_output = generate_inception_module(temp_output, n_inception, n_depth, n_width, l2_weight)
                #temp_output = add_layers(temp_output, 1, n_width, l2_weight)
            else:
                temp_output = add_layers(input_layer, n_depth, n_width, l2_weight)

        if thresholded_output:
            output_layer = Dense(self.n_outputs)(temp_output)
            # output_layer = Dense(1,init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight),bias=True)(temp_output)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(self.n_outputs, init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight), bias=True,
                                 name=name + '_out')(temp_output)

            merged_output = output_layer
            aux_input = input_layer

        return aux_input, input_layer, merged_output, output_layer