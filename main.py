

import model_validation.Validate as Validate
from Datasets import SimulatedData,Gjoa,GJOA2
#import Models.NeuralNetworks.NET1 as NN1
from sklearn.preprocessing import StandardScaler
import pandas as pd

if __name__=='__main__':




    GjoaGAS = Gjoa.fetch_gjoa_data()
    GjoaOIL=GJOA2.fetch_gjoa_data()

    #GjoaOIL.merge(GjoaGAS.X,GjoaGAS.Y)
    #exit()

    #SimData = SimulatedData.fetchSimulatedData()

    #print('\n \n')


    #print(chkInputs.columns)

    Validate.validate_train_test_split(GjoaOIL)

    #Validate.validateRepeat(GjoaData)
    #Validate.validateCV(GjoaData)
    #NN1.SSNET1()

