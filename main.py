

import model_validation.Validate as Validate
from Datasets import SimulatedData,Gjoa,GJOA2

if __name__=='__main__':



    GjoaGAS = Gjoa.fetch_gjoa_data()
    GjoaOIL=GJOA2.fetch_gjoa_data()
    #GjoaOIL=GjoaGAS
    #GjoaOIL.merge(GjoaGAS.X,GjoaGAS.Y)
    #exit()

    #GjoaGAS = SimulatedData.fetchSimulatedData()
    #exit()



    Validate.validate(GjoaOIL,GjoaGAS)
    #Validate.validateCV(GjoaData)
    #NN1.SSNET1()

