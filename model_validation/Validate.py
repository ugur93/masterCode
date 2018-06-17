
import Models.NeuralNetworks.NET1 as NN1
import Models.NeuralNetworks.NN_external as NNE
from Models.NeuralNetworks import NET2_PRESSURE,NN_from_file,NET3,NCNET_CHKPRES,NET_MISC,NCNET1_GJOA2,NCNET_VANILLA_GJOA2,CNN_test,test_model,NCNET4_combined
import time


from .base import *
from .Visualize import *

#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #          'learning_rate': 0.01, 'loss': 'ls'}
    #model=ensemble.GradientBoostingRegressor(**params)
    #model = SVR(C=1000, gamma=0.001,epsilon=0.0001)
MODEL_SAVEFILE_NAME='SSNET2_PRETRAINING_2'




DATA_TYPE='GAsS'
def validate(DataOIL,DataGAS):

    if DATA_TYPE=='GAS':
        Data=DataGAS
    else:
        Data=DataOIL
    #bagging_test(Data)
    validate_train_test_split(Data)
    #ensemble_learning_bagging(Data)
    #grid_searchCV(Data)
    #validateRepeat(Data)
    #search_params = {'n_depth': 2, 'n_width': 100,
    #                 'l2w': 0.0002, 'seed': 3014, 'DATA': 'GAS'}
    #validateCV(Data,params=search_params,save=True)

def validate_train_test_split(Data):
    X = Data.X_transformed#[1:-1]#[500:-1]
    Y = Data.Y_transformed#[1:-1]#[500:-1]


    #subsample(X,Y)

    #X_old,Y_old,X_new,Y_new=split_data(X,Y,split_size=0.3)

    #print(X_old.index)
    #print(X_new.index)
    #X=X[200:-1]
    #Y=Y[200:-1]


    #X_train, Y_train, X_val, Y_val, X_test, Y_test=get_train_test_val_data(X,Y,test_size=0.0,val_size=0.05)

    #print(len(X_train))
    #X_new=X_test
    #Y_new=Y_test
    #X_val=X_train
    #Y_val=Y_train

    #print(X_train.index)
    #print(X_new.index)
    #X_train=X_old
    #Y_train=Y_old
    #X_val=X_train
    #Y_val=Y_train
    PATH='Models/NeuralNetworks/SavedModels2/Weights/GJOA_OIL_WELLS_mae_D2_W20_L20.001_DPR0.h5'
    #PATH = 'Models/NeuralNetworks/SavedModels2/hdf5_files/NCNET_GAS_PRETRAINED_WITH_OLD_DATA'3
    #GJOA QGAS
    PATHS=[]
    for i in range(15):
        if i not in [0,16]:
            PATHS.append('Models/NeuralNetworks/SavedModels2/weights/ENSEMBLE_LEARNING_GAS_WELLS_QGAS_'+str(i)+'.h5')
    print(PATHS)
    #exit()
    #pressure_weights=
    if DATA_TYPE=='GASss':
        PATH = 'Models/NeuralNetworks/SavedModels2/Weights/'
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0, val_size=0.1)

        model = NET2_PRESSURE.SSNET2()

        model.initialize_chk_thresholds(Data, True)
        validateCV(model, model.get_weights(), Data, save=True,filename='GJOA_GAS_WELLS_GAS_FINAL_TEST_2x100_MAPE_FINAL_2')
        exit()

        # model.initialize_zero_thresholds(Data)
        start = time.time()
        print(model.get_config())
        #model.model.load_weights(PATH + 'GJOA_GAS_WELLS_QGAS_HUBER_MODEL_FINAL.h5', by_name=True)
        # print(model.model.get_config())
        model.fit(X_train,Y_train,X_val,Y_val)

        # Fit with old data
        #model.update_model()
        #model.fit(X_train, Y_train, X_val, Y_val)
        #X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.0, val_size=0.2)
    else:
        #GJOA_QOIL
        #pass
        PRESSURE_TAG = 'PDC'
        PATH = 'Models/NeuralNetworks/SavedModels2/Weights/'
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0, val_size=0.1)

        #X=resample(X_train,10)

        #print(X.shape)
        #print(X_train.shape)
        #exit()
        #model = NET2_PRESSURE.VANILLANET()

        #model=NCNET1_GJOA2.NCNET1_GJOA2(DATA='GAS')
        #model = NET2_PRESSURE.SSNET2()

        #model.fit(X_train, Y_train, X_val, Y_val)


        #model=NCNET1_GJOA2.ENSEMBLE(PATHS)
        #model.model.load_weights(PATH+'GJOA_OIL_WELLS_GAS_MODEL22.h5')
        #model = NCNET1_GJOA2.ENSEMBLE(PATHS)
        #model=NCNET_VANILLA_GJOA2.NCNET_VANILLA()
        #model=CNN_test.CNN_GJOAOIL()
        #model = NCNET_CHKPRES.PRESSURE_PBH()


        #model = NCNET_CHKPRES.PRESSURE(tag='PWH')

        #model = NCNET_CHKPRES.PRESSURE_DELTA(tag='PDC',data='OIL')
        #model.model.load_weights(PATH,by_name=True)
        #model = test_model.Test_model()
        #model=NCNET4_combined.NET4_W_PRESSURE2(PATH)

#        model=NCNET4_combined.NET4_W_PRESSURE3()

        #x_last=X_test[-1]

        #y_last = X_test[-1]
        #model.model.load_weights(PATH + 'GJOA_OIL_WELLS_PBH_ALL_DATA.h5', by_name=True)
        #model.model.load_weights(PATH + 'GJOA_OIL_WELLS_PDC_ALL_DATA.h5', by_name=True)
        #model.model.load_weights(PATH+'GJOA_deltaNET_OIL_WELLS_PWH_ALL_DATA.h5',by_name=True)
        #model.model.load_weights(PATH + 'GJOA_OIL_WELLS_GAS_HUBER_MODEL_FINAL2_TESTDATA2.h5', by_name=True)
       # exit()

        # model.initialize_zero_thresholds(Data)
        model.initialize_chk_thresholds(Data, True)
        print(model.get_config())
        start = time.time()

        if False:
            #model.update_model(activation='linear', epoch=1)
            #model.fit(X_train, Y_train, X_val, Y_val)
            #641
            #model.update_model(activation='relu', epoch=141)
            model.fit(X_train, Y_train, X_val, Y_val)
            #model.save_model_to_file(model.model_name)
            #exit()

        elif False:
            validateCV(model, model.get_weights(), Data, save=True,filename=model.model_name)
            exit()
        elif True:
            model.model.load_weights(PATH + 'GJOA_deltaNET_OIL_WELLS_PDC_ALL_DATA.h5', by_name=True)
            model.model.load_weights(PATH + 'GJOA_deltaNET_OIL_WELLS_PWH_ALL_DATA.h5', by_name=True)
            model.model.load_weights(PATH+'GJOA_deltaNET_OIL_WELLS_PBH_ALL_DATA.h5',by_name=True)
            model.model.load_weights(PATH + 'GJOA_OIL_WELLS_GAS_HUBER_MODEL_FINAL2_TESTDATA2.h5', by_name=True)
        else:
            pass

       # model.model.load_weights(PATH + model.model_name+'.h5', by_name=True)

    end = time.time()
    print('Fitted with time: {}'.format(end - start))
    scores, scores_latex = evaluate_model2(model, Data, X_train, X_val, Y_train, Y_val)
    scores=evaluate_model(model, Data, X_train, X_val, Y_train, Y_val)
    print(scores)
    with_line_plot=True
    with_separate_plot=True
    save_fig=False
    PATH='C:/Users/ugurac/Documents/GITFOLDERS/Masteroppgave-2017/figures/Results/NCNET3/'
    file_tag_name='OIL_WELLS_QOIL_'

   # exit()
    #model.save_model_config(scores)

    #plt.scatter(Y_train['GJOA_OIL_SUM_QGAS'],model.predict(X_train)['GJOA_OIL_SUM_QGAS'])
    #plt.plot(np.linspace(0,np.max(Y_train['GJOA_OIL_SUM_QGAS']),10),np.linspace(0,np.max(model.predict(X_train)['GJOA_OIL_SUM_QGAS']),10),color='red')
    #plt.show()
    #get_choke_diff_deviation(model, Data, X_train, Y_train)
    #plot_history(model)
    #plot_pressure_model(model, Data, X,Y,X_train, X_val, Y_train, Y_val, with_line_plot=False,
    #                        with_separate_plot=False)
    #plt.show()
    if False:
        #import seaborn
        tag = model.tag

        Y_p=model.predict(X).set_index(X.index)
       #Y_p_test=model.predict(X_val).set_index(X_val.index)


        for name in model.well_names:
            fig,axes=plt.subplots(3,1,sharex=True)
            axes=axes.flatten()
            #+Data.inverse_transform(X, 'X')['C1_shifted_PWH']
            axes[0].grid()
            axes[1].grid()
            axes[2].grid()
            axes[0].axvline(len(X_train)+X_train.index[0],-20,20)
            axes[1].axvline(len(X_train)+X_train.index[0], -20, 20)
            axes[2].axvline(len(X_train)+X_train.index[0], -20, 20)


            axes[0].plot(Data.inverse_transform(Y_p,'Y')[name+'_'+tag], marker='.',color='black',label=name+'_'+tag+'_predicted')
            axes[0].plot(Data.inverse_transform(Y,'Y')[name+'_'+tag], marker='.',color='blue',label=name+'_'+tag)
            #axes[0].plot(Data.inverse_transform(Y_p_test, 'Y')[name + '_' + tag], marker='.', color='green',
            #             label=name + '_' + tag + '_predicted')
            #axes[0].plot(Data.inverse_transform(Y_val, 'Y')[name + '_' + tag], marker='.', color='red',
            #             label=name + '_' + tag)
            axes[0].legend()

            if model.type=='DELTA':
                axes[1].plot(Data.inverse_transform(X, 'X')[name + '_CHK'], '-.', label=name + '_CHK')
                axes[1].plot(Data.inverse_transform(X, 'X')[name + '_shifted_CHK'] * -1, color='red',
                             label=name + '_shifted_CHK')
                for key in model.chk_names:
                    axes[1].plot(Data.inverse_transform(X,'X')[key+'_delta_CHK'],label=key+'_delta_CHK')
                if DATA_TYPE!='GAS' and model.tag=='PDC':
                    axes[1].plot(Data.inverse_transform(X, 'X')['GJOA_RISER_delta_CHK'] ,label='GJOA_RISER_delta_CHK')
                axes[1].legend()
                axes[2].plot(Data.inverse_transform(X, 'X')[name + '_shifted_' + tag], color='red',
                             label=name + '_prev_' + tag)
                axes[2].plot(Data.inverse_transform(X, 'X')[name + '_' + tag], color='blue', label=name + '_now_' + tag)
                axes[2].plot(Data.inverse_transform(Y, 'Y')[name + '_delta_' + tag], color='green',
                             label=name + '_delta_' + tag)
                axes[2].plot(Data.inverse_transform(Y_p, 'Y')[name + '_delta_' + tag],color='orange',
                             label=name + '_delta_' + tag + '_predicted')
                plt.legend()
            else:
                for key in model.chk_names:
                    axes[1].plot(Data.inverse_transform(X,'X')[key+'_CHK'],label=key+'_CHK')
                if DATA_TYPE!='GAS' and model.tag=='PDC':
                    axes[1].plot(Data.inverse_transform(X, 'X')['GJOA_RISER_OIL_B_CHK'], label='GJOA_RISER_OIL_B_CHK')
                axes[1].legend()

            if False:
                if model.type=='DELTA':
                    fig,axes=plt.subplots(1,1)
                    axes.scatter(Data.inverse_transform(X,'X')[name+'_delta_'+'CHK'],Data.inverse_transform(Y_p,'Y')[name+'_delta_'+tag],color='red',label='True')
                    axes.scatter(Data.inverse_transform(X, 'X')[name + '_delta_' + 'CHK'],
                                 Data.inverse_transform(Y, 'Y')[name + '_delta_' + tag],label='predicted',color='blue')
                    axes.set_xlabel(name+'_delta_'+'CHK')
                    axes.set_ylabel(name+'_delta_'+tag)
                    plt.legend()
                else:
                    fig, axes = plt.subplots(1, 1)
                    axes.scatter(Data.inverse_transform(X, 'X')[name + '_' + 'CHK'],
                                 Data.inverse_transform(Y_p, 'Y')[name + '_' + tag], color='red', label='True')
                    axes.scatter(Data.inverse_transform(X, 'X')[name + '_' + 'CHK'],
                                 Data.inverse_transform(Y, 'Y')[name + '_' + tag], label='predicted',
                                 color='blue')
                    axes.set_xlabel(name + '_delta_' + 'CHK')
                    axes.set_ylabel(name + '_delta_' + tag)
                    plt.legend()

    else:
        visualize(model, Data, X_train, X_val, Y_train, Y_val, output_cols=[], input_cols=[],with_line_plot=with_line_plot,
                  with_separate_plot=with_separate_plot,save_fig=save_fig,PATH=PATH,file_tag_name=file_tag_name)

    plt.show()




def train_and_evaluate(model,Data,X_train,X_val,Y_train,Y_val):


    #model.update_model(activation='linear',epoch=1)
    #model.fit(X_train,Y_train,X_val,Y_val)

    #model.update_model(activation='relu',epoch=641-500)
    #model.fit(X_train, Y_train, X_val, Y_val)
    PATH = 'Models/NeuralNetworks/SavedModels2/Weights/'

    model.model.load_weights(PATH + model.model_name + '.h5', by_name=True)

    return evaluate_model(model, Data, X_train, X_val, Y_train, Y_val)

def results_to_latex(RMSE_TRAIN,RMSE_TEST,R2_TRAIN,R2_TEST,MAPE_TRAIN,MAPE_TEST,well_names):
    temp_s='\n'
    well_names2 = ['C1', 'C2', 'C3', 'C4', 'B1', 'B3', 'D1']
    KEY_MAP = {}
    for i, key in zip(range(1, len(well_names2) + 1), well_names2):
        KEY_MAP[key + '_QGAS'] = 'Well O{}'.format(i)
        KEY_MAP[key + '_QOIL'] = 'Well O{}'.format(i)
        KEY_MAP[key + '_PDC'] = 'Well O{}'.format(i)
        KEY_MAP[key + '_PBH'] = 'Well O{}'.format(i)
        KEY_MAP[key + '_PWH'] = 'Well O{}'.format(i)
    well_names2 = ['F1', 'B2', 'D3', 'E1']
    for i, key in zip(range(1, len(well_names2) + 1), well_names2):
        KEY_MAP[key + '_QGAS'] = 'Well G{}'.format(i)
    KEY_MAP['GJOA_OIL_SUM_QGAS'] = 'Total production flow rate'
    KEY_MAP['GJOA_TOTAL_SUM_QOIL'] = 'Total production flow rate'
    KEY_MAP['GJOA_QGAS'] = 'Total production flow rate'

    # print(KEY_MAP)
    # exit()
    for key in ['A', 'B', 'C', 'D']:
        KEY_MAP[key + '_QGAS'] = 'Well ' + key
    R2_TRAIN=R2_TRAIN*100
    R2_TEST = R2_TEST
    for key in well_names:
        temp_s+='{0:s} & {1:0.2f} & {2:0.2f} & {3:0.2f}\\% \\\\ \n '.format(KEY_MAP[key],RMSE_TEST[key],R2_TEST[key],MAPE_TEST[key])
    return temp_s

def validateCV(model,init_weights,Data,params=None,save=True,filename=''):
    X = Data.X_transformed
    Y = Data.Y_transformed


    X, Y, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0, val_size=0.1)

   # X.drop([0], inplace=True)
    #Y.drop([0], inplace=True)


    kfold=model_selection.KFold(n_splits=5,random_state=False)

    scores_rmse_train=None
    scores_r2_test = None
    scores_rmse_test = None
    scores_r2_train = None
    scores_mape_train  =None
    scores_mape_test = None



    #zprint scores
    #print(chkInputs.index)
    i=1

    #filename='GJOA_OIL_WELLS_GAS_FINAL_TEST2'



    for train_index,test_index in kfold.split(X.index):

        model.set_weights(init_weights)

        print(test_index)
        if False:
            X_train=X.iloc[train_index]
            X_val=X.iloc[test_index]
            Y_train=Y.iloc[train_index]
            Y_val=Y.iloc[test_index]
        else:
            X_train=X
            Y_train=Y
            X_val=X_val
            Y_val=Y_val


        scores=train_and_evaluate(model,Data,X_train,X_val,Y_train,Y_val)

        if scores_rmse_train is None:
            scores_rmse_train=scores['RMSE_train'].to_frame().T
            scores_rmse_test = scores['RMSE_test'].to_frame().T
            scores_r2_train = scores['R2_train'].to_frame().T
            scores_r2_test = scores['R2_test'].to_frame().T
            scores_mape_train = scores['MAPE_train'].to_frame().T
            scores_mape_test = scores['MAPE_test'].to_frame().T
        else:
            scores_rmse_train=scores_rmse_train.append(scores['RMSE_train'].to_frame().T)
            scores_rmse_test=scores_rmse_test.append(scores['RMSE_test'].to_frame().T)
            scores_r2_train=scores_r2_train.append(scores['R2_train'].to_frame().T)
            scores_r2_test=scores_r2_test.append(scores['R2_test'].to_frame().T)
            scores_mape_train = scores_mape_train.append(scores['MAPE_train'].to_frame().T)
            scores_mape_test = scores_mape_test.append(scores['MAPE_test'].to_frame().T)
        print('--------------------------------')

        print('RMSE TEST')
        print(scores_rmse_test)
        print('MAPE TEST')
        print(scores_mape_test)
        break


    scores_rmse_train.set_index(pd.Index(range(0,len(scores_rmse_train))))
    scores_rmse_test.set_index(pd.Index(range(0, len(scores_rmse_test))))
    scores_r2_train.set_index(pd.Index(range(0, len(scores_r2_train))))
    scores_r2_test.set_index(pd.Index(range(0, len(scores_r2_test))))
    scores_mape_train.set_index(pd.Index(range(0, len(scores_mape_train))))
    scores_mape_test.set_index(pd.Index(range(0, len(scores_mape_test))))

    RMSE_TRAIN = np.mean(scores_rmse_train)
    RMSE_TEST = np.mean(scores_rmse_test)
    R2_TRAIN = np.mean(scores_r2_train)
    R2_TEST = np.mean(scores_r2_test)
    MAPE_TRAIN = np.mean(scores_mape_train)
    MAPE_TEST = np.mean(scores_mape_test)

    print(MAPE_TRAIN)
    print(MAPE_TEST)


    if save:
        s='RMSE_TRAIN: \n{}\n'.format(RMSE_TRAIN)
        s += 'RMSE_TEST: \n{}\n'.format(RMSE_TEST)
        s += 'R2_TRAIN: \n{}\n'.format(R2_TRAIN)
        s += 'R2_TEST: \n{}\n'.format(R2_TEST)
        s += 'sMAPE_TRAIN: \n{}\n'.format(MAPE_TRAIN)
        s += 'sMAPE_TEST: \n{}\n'.format(MAPE_TEST)
        s += 'RMSE_TRAIN_SCORES: \n{}\n'.format(scores_rmse_train)
        s += 'RMSE_TEST_SCORES: \n{}\n'.format(scores_rmse_test)

        s += 'sMAPE_TRAIN_SCORES: \n{}\n'.format(scores_mape_train)
        s += 'sMAPE_TEST_SCORES: \n{}\n'.format(scores_mape_test)


        s+=results_to_latex(RMSE_TRAIN,RMSE_TEST,R2_TRAIN,R2_TEST,MAPE_TRAIN,MAPE_TEST,model.output_tag_ordered_list)



        PATH = 'Models/NeuralNetworks/CV_results/' + filename
        f = open(PATH, 'w')
        f.write(s)
        f.close()
    #exit()
    return {'RMSE_train':RMSE_TRAIN,'RMSE_test':RMSE_TEST,'R2_train':R2_TRAIN,'R2_test':R2_TEST}





def plotTrainingHistory(model):
    history = model.get_history()
    plt.figure()
    plt.plot(history)
    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()



def generate_grid(search_params):
    from itertools import product
    items=sorted(search_params.items())
    keys, values=zip(*items)
    params=[]
    for v in product(*values):
        params.append(dict(zip(keys,v)))
    return params
def grid_searchCV(Data):

    L2_WEIGHTS=np.concatenate((np.arange(0.0001,0.001,0.00005),np.arange(0.001,0.003,0.0001)))
    search_params={'n_depth':[2],'n_width':[20,30,50,60,80,90,100],
                   'l2w':np.arange(0.0001,0.0005,0.00005),'seed':[3014]}#,'DATA':['GAS']}
    print(search_params['l2w'])
    #search_params = {'n_depth': [2], 'n_width': [50],
    #                 'l2w':[0.0001], 'seed': [3014]}

    grid_params=generate_grid(search_params)

    len_grid=len(grid_params)

    print('Size of search space: {}'.format(len_grid))


    best_results=None
    best_cost=1e100
    best_params={}

    PRESSURE_TYPE='PBH'
    filename='GRID_SEARCH_DeltaNET_'+PRESSURE_TYPE
    #filename = 'GRID_SEARCH_GAS_WELLS_QGAS_DEPTH_2_TEST_6'
    ii=1
    pd.options.display.float_format = '{:.2f}'.format
    #col_eval=['GJOA_OIL_SUM_QGAS']
    #col_eval=['GJOA_TOTAL_SUM_QOIL']
    col_eval=['GJOA_QGAS']
    for params in grid_params:
        params['seed'] = int(params['seed'])
        print('Training with params: {}, filename: {} '.format(params, filename))
        print('\n\n\n')
        print('On n_grid: {} of {}'.format(ii, len_grid))

        ii+=1

        #model=NCNET1_GJOA2.NCNET1_GJOA2(**params,output_act='relu',n_epoch=10000)
        #model = NET2_PRESSURE.SSNET2(**params, output_act='relu', n_epoch=10000)

        model = NCNET_CHKPRES.PRESSURE_DELTA(tag=PRESSURE_TYPE,n_epoch=10000)

        #model = NCNET_CHKPRES.PRESSURE_DELTA(tag=PRESSURE_TYPE,data='OIL',n_epoch=10000)
        model.initialize_chk_thresholds(Data, True)

        init_weights=model.get_weights()

        scores=validateCV(model,init_weights,Data,params,save=False)

        #current_cost =scores['RMSE_test'][col_eval].values
        current_cost = np.sum(scores['RMSE_test'])
        #print(np.sqrt(scores['MSE_test']))
        #print(current_cost)
        if current_cost<best_cost:
            best_cost=current_cost
            best_params=params
            best_results=scores


        print('THIS COST: {}, BEST COST: {}'.format(current_cost, best_cost))
        print('Best params:{} \n'.format(best_params))
        print('BEST SCORES: ')
        print(best_results)
        del model


    s='Best results: \n'
    s+='Best params:{} \n'.format(best_params)
    s+='Best COST: {} \n'.format(best_cost)
    s+='BEST SCORES: \n {}'.format(best_results)

    PATH = 'Models/NeuralNetworks/'+filename
    f = open(PATH, 'w')
    f.write(s)
    f.close()
def grid_search2(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed

    cum_thresh=15

    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)

    search_params={'n_depth':[2],'n_width':[30,40,50,60,70,80,90,100],
                   'l2w':np.arange(0.001,0.003,0.0001),'seed':[3014]}

    #search_params = {'n_depth': [2], 'n_width': [50],
    #                 'l2w':[0.0001], 'seed': [3014]}

    grid_params=generate_grid(search_params)

    len_grid=len(grid_params)

    print('Size of search space: {}'.format(len_grid))


    best_results=None
    best_cost=1e100
    best_params={}

    PRESSURE_TYPE='PWH'
    #filename='GRID_SEARCH_OIL_WELLS_OIL2'
    filename = 'GRID_SEARCH_GAS_WELLS_GAS_2_REGHIGH'
    ii=1
    pd.options.display.float_format = '{:.2f}'.format
    #col_eval=['GJOA_TOTAL_SUM_QOIL']
    col_eval = ['GJOA_QGAS']
    for params in grid_params:

        print('Training with params: {}, filename: {} '.format(params, filename))
        params['seed']=int(params['seed'])
        #model = NCNET1_GJOA2.NCNET1_GJOA2(**params)
        #model=NCNET_CHKPRES.PRESSURE(**params,tag=PRESSURE_TYPE)
        #model = NCNET_CHKPRES.PRESSURE_PDC(**params)
        model = NET2_PRESSURE.SSNET2(**params)
        print('\n\n\n')
        print('On n_grid: {} of {}'.format(ii,len_grid))
        ii+=1
        model.initialize_chk_thresholds(Data, True)
        model.fit(X_train, Y_train, X_val, Y_val)
        scores= evaluate_model(model,Data, X_train, X_val, Y_train, Y_val)
        #print(scores)
        current_cost =scores['RMSE_test'][col_eval].values
        #current_cost = np.sum(scores['RMSE_test'])
        #print(np.sqrt(scores['MSE_test']))
        #print(current_cost)
        if current_cost<best_cost:
            best_cost=current_cost
            best_params=params
            best_results=scores
        del model

        print('THIS COST: {}, BEST COST: {}'.format(current_cost, best_cost))
        print('Best params:{} \n'.format(best_params))
        print('BEST SCORES: ')
        print(best_results)



    s='Best results: \n'
    s+='Best params:{} \n'.format(best_params)
    s+='Best COST: {} \n'.format(best_cost)
    s+='BEST SCORES: \n {}'.format(best_results)

    PATH = 'Models/NeuralNetworks/'+filename
    f = open(PATH, 'w')
    f.write(s)
    f.close()

def ensemble_learning(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed

    cum_thresh = 15

    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)

    search_params = {'n_depth': [2], 'n_width': [100],
                     'l2w': [0.0003], 'seed': np.random.randint(100,10000,15)}
    grid_params = generate_grid(search_params)

    len_grid = len(grid_params)

    print('Size of search space: {}'.format(len_grid))


    name='ENSEMBLE_LEARNING_3_GAS_'
    PATH='Models/NeuralNetworks/SavedModels2/hdf5_files/'
    PATHS=[]
    i=0
    for params in grid_params:
        params['seed'] = int(params['seed'])
        model = NCNET1_GJOA2.NCNET1_GJOA2(**params)
        model.model_name=name+str(i)
        PATHS.append(PATH+model.model_name+'.h5')
        i+=1

        print('Training with params: {}'.format(params))
        model.initialize_chk_thresholds(Data, True)
        #model.fit(X_train, Y_train, X_val, Y_val)
        #model.update_model()
        model.fit(X_train, Y_train, X_val, Y_val)

        scores, scores_latex = evaluate_model(model, Data, X_train, X_val, Y_train, Y_val)
        model.save_model_to_file(model.model_name, scores)
        print(scores)


        del model

    print('[')
    for path in PATHS:
        print(path + ',')
    print(']')




def bagging_test(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed

    cum_thresh = 15

    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)
    search_params = {'n_depth': [4], 'n_width': [50],
                     'l2w': [0.000001], 'seed': np.random.randint(1, 10000, 10)}

    params={'n_depth': 2, 'n_width': 50,
                     'l2w': 0.0001, 'seed': 9035}
    model = NCNET1_GJOA2.NCNET1_GJOA2(**params)

    #tags=model.output_tag_ordered_list

    mod=ensemble.BaggingRegressor(base_estimator=model,n_estimators=2)
    print(X_train.shape)
    mod.fit(X_train,Y_train['GJOA_OIL_QGAS'])

    print(mod.predict(X_val))

def ensemble_learning_bagging(Data):
    X = Data.X_transformed
    Y = Data.Y_transformed
    GS_SIZE=15

    print('Grid Search size: {}'.format(GS_SIZE))
    params = {'n_depth': 2, 'n_width': 60,
              'l2w': 0.0001, 'seed': 3014}

    name='ENSEMBLE_LEARNING_GAS_WELLS_QGAS_'
    PATH='Models/NeuralNetworks/SavedModels2/hdf5_files/'
    PATHS=[]
    i=1
    for i in range(GS_SIZE):
        if i>=0:
            print('Training with params: {}'.format(params))

            #model = NCNET1_GJOA2.NCNET1_GJOA2(**params)
            model = NET2_PRESSURE.SSNET2(**params)

            model.model_name=name+str(i)

            PATHS.append(PATH+model.model_name+'.h5')

            X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)
            X_train,Y_train=subsample(X_train,Y_train)


            model.initialize_chk_thresholds(Data, True)
            model.fit(X_train, Y_train, X_val, Y_val)

            scores, scores_latex = evaluate_model2(model, Data, X_train, X_val, Y_train, Y_val)
            model.save_model_to_file(model.model_name)
            print(scores)
            del model
        else:
            X_train, Y_train, X_val, Y_val, X_test, Y_test = get_train_test_val_data(X, Y, test_size=0.1, val_size=0.2)
            X_train, Y_train = subsample(X_train, Y_train)



    print('[')
    for path in PATHS:
        print(path+',')
    print(']')