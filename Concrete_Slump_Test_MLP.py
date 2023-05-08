from sklearn import neural_network
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# INCARCARE DATE
data = pd.read_csv('slump_test.data')

date = np.array(data.drop(['No','SLUMP(cm)','FLOW(cm)','Compressive Strength (28-day)(Mpa)'],axis=1)) 
etichete= np.array(data.drop(['No','Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.'],axis=1)) 

print(etichete)
size = len(data['No'])
train_size = int(0.75*size)
test_size=int(0.25*size)

#IMPARTIRE IN TRAIN SI TEST
date_train = date[:train_size,:]
date_test = date[train_size:,:]

etichete_train = etichete[:train_size,:]
etichete_test = etichete[train_size:,:]

# CREARE SI ANTRENARE MLP
regr = neural_network.MLPRegressor(learning_rate_init=0.01, max_iter=2000, hidden_layer_sizes=(7))
regr.fit(date_train,etichete_train)

# TESTARE MLP
predictii = regr.predict(date_test)

# EROARE  
MSE = mean_squared_error(predictii,etichete_test)
print('\nMSE: ', str(MSE))   
