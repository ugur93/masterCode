
from .base import *
from .base_class import NN_BASE



#GOOD

#incept 3
# dept 1
# width 20
#l2 0.0001
#opt rmsprop
class NCNET1_GJOA2(NN_BASE):


    def __init__(self):

        name='NCNET1_2'


        self.n_inputs=3
        self.n_outputs=8
        self.SCALE=100

        # Input module config
        self.n_inception = 0 #(n_inception, n_depth inception)
        self.n_depth = 2
        self.n_width = 20
        self.l2weight = 0.0001
        self.add_thresholded_output=True

        self.output_tags = {
            'C1_out': ['C1_QOIL'],
            'C2_out': ['C2_QOIL'],
            'C3_out': ['C3_QOIL'],
            'C4_out': ['C4_QOIL'],
            'D1_out': ['D1_QOIL'],
            'B3_out': ['B3_QOIL'],
            'B1_out': ['B1_QOIL'],
            'GJOA_TOTAL': ['GJOA_TOTAL_QOIL_SUM']

        }


        self.input_tags={
            'C1':['C1_CHK','C1_PDC','C1_PWH'],
            'C2':['C2_CHK','C2_PDC','C2_PWH'],
            'C3':['C3_CHK','C3_PDC','C3_PWH'],
            'C4':['C4_CHK','C4_PDC','C4_PWH'],
            'D1':['D1_CHK','D1_PDC','D1_PWH'],
            'B3':['B3_CHK','B3_PDC','B3_PWH'],
            'B1':['B1_CHK','B1_PDC','B1_PWH']
        }

        #self.input_tags=['C1','C2','C3','C4','D1','B3','B1']

        self.loss_weights=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]

        #Training config
        optimizer = 'adam' #SGD(momentum=0.9,nesterov=True)
        loss = 'mse'
        nb_epoch = 5000
        batch_size = 1000
        verbose = 0

        train_params={'optimizer':optimizer,'loss':loss,'nb_epoch':nb_epoch,'batch_size':batch_size,'verbose':verbose}

        super().__init__(name,train_params)

    def initialize_model(self):
        print('Initializing %s' % (self.model_name))


        for key in self.input_tags.keys():
            aux_input,input,merged_out,out=self.generate_input_module(n_depth=self.n_depth, n_width=self.n_width,
                                                                    n_input=self.n_inputs, n_inception=self.n_inception,
                                                                    l2_weight=self.l2weight, name=key,thresholded_output=self.add_thresholded_output)
            self.aux_inputs.append(aux_input)
            self.inputs.append(input)
            self.merged_outputs.append(merged_out)
            self.outputs.append(out)
            #print(merged_out.name)

        merged_input = merge(self.merged_outputs, mode='sum', name='GJOA_TOTAL')

        self.merged_outputs.append(merged_input)
        #self.merged_outputs=sort_output_layers(self.merged_outputs,self.output_tags)



        #print(self.merged_outputs)
        inputs = self.inputs

        if self.add_thresholded_output:
            inputs+=self.aux_inputs

        self.model = Model(input=inputs, output=self.merged_outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
        #print(self.get_config())
        #print(self.model.summary())

    def generate_input_module(self,n_depth, n_width, l2_weight, name, n_input, thresholded_output, n_inception=0):

        input_layer = Input(shape=(n_input,), dtype='float32', name=name)
        # temp_output=Dropout(0.1)(input_layer)

        if n_depth == 0:
            temp_output = input_layer
        else:
            if n_inception > 1:
                #temp_output = add_layers(input_layer, 1, n_width, l2_weight)
                temp_output = generate_inception_module(input_layer, n_inception, n_depth, n_width, l2_weight)
                temp_output = add_layers(temp_output, 1, n_width*n_inception, l2_weight)
                #temp_output = generate_inception_module(temp_output, n_inception, n_depth, n_width, l2_weight)
                #temp_output = add_layers(temp_output, 1, n_width, l2_weight)
            else:
                temp_output = add_layers(input_layer, n_depth, n_width, l2_weight)

        if thresholded_output:
            output_layer = Dense(1)(temp_output)
            # output_layer = Dense(1,init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight),bias=True)(temp_output)
            aux_input, merged_output = add_thresholded_output(output_layer, n_input, name)
        else:
            output_layer = Dense(1, init=INIT, W_regularizer=l2(l2_weight), b_regularizer=l2(l2_weight), bias=True,
                                 name=name + '_out')(temp_output)

            merged_output = output_layer
            aux_input = input_layer

        return aux_input, input_layer, merged_output, output_layer