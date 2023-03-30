##import module
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import load_model, save_model
from keras.callbacks import ModelCheckpoint
import os
os.chdir("C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/.../MQL5/Include/Hedge_include")
from RFECV import RFECV_RandomForest


##
def neuronalNetwork(Train, data, new_name_file, old_name_file, epoch, day_windows, plot, accuracyBool, RFECV_step, split):
    ##___________________________________________________
    ##___________________Preprocessing___________________
    X_data = data[['tick_volume', 'spread', 'tick_closeHigh', 'tick_closeLow', 'variation_closeHigh', 'variation_closeLow', 'variation_closeOpen', 'volume_profile', 'ADX', 'ADX_PDI', 'ADX_NDI', 'AO', 'ATR', 'BearsPower', 'BullsPower', 'Var_BBP', 'CCI', 'DEMA', 'Var_DEMA', 'Kijun', 'SBB', 'Var_Tenkan', 'Var_Kijun', 'Var_SSB', 'Var_SSA', 'Var_SSBSSA', 'MACD', 'Signal_MACD', 'Momentum', 'RSI','RVI', 'Signal_RVI', 'STOCH', 'Signal_STOCH', 'UO']].values
    Y_data = data['close'].values

    scaler = StandardScaler()
    X_scaled_data = scaler.fit_transform(X_data)
    Y_scaled_data = scaler.fit_transform(Y_data.reshape(-1,1))
    ##___________________________________________________
    ##___________________Split dataset___________________
    if not(Train):
        split=0.01
    else:
        split=0.8

    training_data_len = math.ceil(len(X_data)* split)

    X_train_data = X_scaled_data[: training_data_len, :]
    Y_train_data = Y_scaled_data[: training_data_len, :]
    x_train = []
    y_train = []

    for i in range(day_windows, len(X_train_data)):
        windows_x=[]
        for j in range(1,day_windows+1):
            windows_x.append(X_train_data[i-j])
        x_train.append(windows_x)
        y_train.append(Y_train_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], -1, 1))

    #-------

    Y_test_data = Y_scaled_data[training_data_len: , : ]
    X_test_data = X_scaled_data[training_data_len: , : ]
    y_test = []
    x_test = []

    for i in range(day_windows, len(X_test_data)):
        windows_x=[]
        for j in range(1,day_windows+1):
            windows_x.append(X_test_data[i-j])
        x_test.append(windows_x)
        y_test.append(Y_test_data[i])

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], -1, 1))
    ##___________________________________________________
    ##_________________Features Selection________________
    rfecv=[]
    if RFECV_step:
        print("RFECV procedure :\n")
        rfecv = RFECV_RandomForest(x_test[:int(len(x_test)*split)], y_test[:int(len(y_test)*split)], RFECV_step)
    ##___________________________________________________
    ##_________________Model architecture________________
    if Train and (old_name_file == None or old_name_file == " ") :

        tf.random.set_seed(7) # fix random seed for reproducibility

        model = keras.Sequential()
        model.add(layers.LSTM(200, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(layers.LSTM(125, return_sequences=False))
        model.add(layers.Dense(100))
        model.add(layers.Dense(75))
        model.add(layers.Dense(50))
        model.add(layers.Dense(50))
        model.add(layers.Dense(25))
        model.add(layers.Dense(25))
        model.add(layers.Dense(5))
        model.add(layers.Dense(5))
        model.add(layers.Dense(1))

        """
        model = keras.Sequential()
        model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(layers.LSTM(75, return_sequences=True))
        model.add(layers.LSTM(75, return_sequences=True))
        model.add(layers.LSTM(50, return_sequences=True))
        model.add(layers.LSTM(25, return_sequences=True))
        model.add(layers.LSTM(5, return_sequences=False))
        model.add(layers.Dense(1))
        """
        """
        #Nul sur 2 epoch :
        model = keras.Sequential()
        model.add(layers.LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(units=96, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(units=96, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(units=96))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units=1))
        """
        """
        model = keras.Sequential()
        model.add(layers.GRU(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(layers.GRU(100, return_sequences=False))
        model.add(layers.Dense(100))
        model.add(layers.Dense(75))
        model.add(layers.Dense(75))
        model.add(layers.Dense(50))
        model.add(layers.Dense(50))
        model.add(layers.Dense(25))
        model.add(layers.Dense(25))
        model.add(layers.Dense(5))
        model.add(layers.Dense(5))
        model.add(layers.Dense(1))
        #tester dense avec la fct° d'activat° relu ou autre (sigmoid, tanh) et les autres couches
        """
        """
        #Pue la merde
        model = keras.Sequential()
        model.add(layers.convolutional.Convolution1D(100, kernel_size=10, input_shape=(x_train.shape[1],1)))
        model.add(layers.convolutional.MaxPooling1D(2))
        model.add(layers.convolutional.Convolution1D(50, kernel_size=9))
        model.add(layers.convolutional.MaxPooling1D(2))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(50))
        model.add(layers.Dense(50))
        model.add(layers.Dense(25))
        model.add(layers.Dense(25))
        model.add(layers.Dropout(0.2))
        model.add(layers.Activation('tanh'))
        model.add(layers.Dense(1))
        """

        model.summary()
        adam = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer='adam', loss='mse')
    ##___________________________________________________
    ##______________Train, Test & save Model_____________
    print("Fiting model and data procedure :\n")

    #EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.003, patience=2, verbose=1, mode='auto', baseline=0.1)

    if old_name_file != None and old_name_file != " " :
        old_filepath = f'C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/.../MQL5/Include/Hedge_include/saved_best_models/MonoOutput/{old_name_file}'
        model = load_model(old_filepath)

    if Train:
        new_filepath = f'C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/.../MQL5/Include/Hedge_include/saved_best_models/MonoOutput/{new_name_file}'
        model.fit(x_train, y_train, batch_size= 1, epochs=epoch, verbose=1)
        save_model(model,new_filepath)

    predictions_test = model.predict(x_test)
    predictions_train = model.predict(x_train)
    predictions_test = scaler.inverse_transform(predictions_test)
    predictions_train = scaler.inverse_transform(predictions_train)
    ##___________________________________________________
    ##______________________Metrics______________________

    y_test = scaler.inverse_transform(y_test)
    y_train = scaler.inverse_transform(y_train)

    def pourcentageAccuracy(y_test,y_pred):
        accuracy=0

        for i in range(len(y_test)):
            accuracy+=100*math.exp(min(y_pred[i][0],y_test[i][0])/max(y_pred[i][0],y_test[i][0])-1)**1000

        accuracy/=len(y_test)

        return accuracy

    accuracy_test = pourcentageAccuracy(y_test,predictions_test)
    accuracy_train = pourcentageAccuracy(y_train,predictions_train)

    print("accuracy test = ", accuracy_test)
    print("accuracy train = ", accuracy_train)

    rmse = np.sqrt(np.mean(predictions_test - y_test)**2)
    print("rmse = ", rmse)
    ##___________________________________________________
    ##_________________Print Predictions_________________
    if plot:

        """predictions_test = 5347.2202264094*np.array(data.filter(['close'])[training_data_len+day_windows-1:-6])*np.array(predictions_test)
        predictions_train = 5347.2202264094*np.array(data.filter(['close'])[day_windows-1:training_data_len-6])*np.array(predictions_train)"""


        Close_Train = data.filter(['close'])[day_windows:training_data_len]
        Close_Train['Predictions Train'] = np.reshape(predictions_train,(-1))
        Close_Test = data.filter(['close'])[training_data_len+day_windows:]
        Close_Test['Predictions Test'] = np.reshape(predictions_test,(-1))

        plt.figure(figsize=(12,6))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')

        plt.plot(Close_Train[['close', 'Predictions Train']])
        plt.plot(Close_Test[['close', 'Predictions Test']])
        plt.legend(['Train Price', 'Predictions Train','Test Price', 'Predictions Test'], loc='lower right')

        plt.grid()
        plt.show()

    if accuracyBool:
        return accuracy_test, rfecv
    return rmse, rfecv

##________________________________________________________________________________________________
##________________________________________________________________________________________________
##||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##________________________________________________________________________________________________
##________________________________________________________________________________________________

#1
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-1.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures1, rfecv = neuronalNetwork(Train=True,data=data, new_name_file="EURUSD_2j-43Features-1",old_name_file=None, epoch=3, day_windows=5, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#2
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-2.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures2, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-2",old_name_file="EURUSD_2j-43Features-1", epoch=3, day_windows=5, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#3
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-3.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures3, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-3",old_name_file="EURUSD_2j-43Features-2", epoch=3, day_windows=5, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#4
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-4.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures4, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-4",old_name_file="EURUSD_2j-43Features-3", epoch=3, day_windows=5, plot=True, accuracyBool=True, RFECV_step=0, split=0.1)
"""
#5
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-5.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures5, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-5",old_name_file="EURUSD_2j-43Features-4", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#6
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-6.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures6, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-6",old_name_file="EURUSD_2j-43Features-5", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#7
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-7.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures7, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-7",old_name_file="EURUSD_2j-43Features-6", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#8
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-8.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures8, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-8",old_name_file="EURUSD_2j-43Features-7", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#9
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-9.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures9, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-9",old_name_file="EURUSD_2j-43Features-8", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#10
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-10.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures10, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-10",old_name_file="EURUSD_2j-43Features-9", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#11
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-11.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures11, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-11",old_name_file="EURUSD_2j-43Features-10", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#12
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-12.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures12, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-12",old_name_file="EURUSD_2j-43Features-11", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#13
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-13.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures13, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-13",old_name_file="EURUSD_2j-43Features-12", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#14
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-14.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures14, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-14",old_name_file="EURUSD_2j-43Features-13", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#15
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-15.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures15, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-43Features-15",old_name_file="EURUSD_2j-43Features-14", epoch=3, day_windows=3, plot=False, accuracyBool=True, RFECV_step=0, split=0.1)

#Test
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-test.csv',encoding = "ISO-8859-1",sep='\t')

data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures_test, rfecv = neuronalNetwork(Train=False, data=data, new_name_file=None,old_name_file="EURUSD_2j-43Features-10", epoch=3, day_windows=3, plot=True, accuracyBool=True, RFECV_step=0, split=1)
"""
"""
print("metrics_allFeatures = ", metrics_allFeatures11,metrics_allFeatures12,metrics_allFeatures13,metrics_allFeatures14,metrics_allFeatures15,metrics_allFeatures_test)

##
##

col=[]
for i in range(5):
    row=[]
    for j in range(42):
        row.append(rfecv[i*42+j])
    col.append(row)
col = np.array(col)
print(col)

BestFeaturesSelection = pd.DataFrame(col)
columns = ['close', 'open', 'high', 'low', 'tick_volume', 'spread', 'tick_closeHigh', 'tick_closeLow', 'variation_closeHigh', 'variation_closeLow', 'variation_closeOpen', 'average_price', 'volume_profile', 'ADX', 'ADX_PDI', 'ADX_NDI', 'AO', 'ATR', 'BearsPower', 'BullsPower', 'Var_BBP', 'CCI', 'DEMA', 'Var_DEMA', 'Tenkan', 'Kijun', 'SBB', 'SSA', 'Var_Tenkan', 'Var_Kijun', 'Var_SSB', 'Var_SSA', 'Var_SSBSSA', 'MACD', 'Signal_MACD', 'Momentum', 'RSI','RVI', 'Signal_RVI', 'STOCH', 'Signal_STOCH', 'UO']

nb=[k for k in range (len(rfecv))]

plt.figure()
legend1=[]
for k in range(BestFeaturesSelection.shape[1]):
    if BestFeaturesSelection[k][0]<=20:
        legend1.append(columns[k])
        plt.plot(BestFeaturesSelection[k])

plt.legend(legend1, loc='lower right')

plt.figure()
legend2=[]
for k in range(BestFeaturesSelection.shape[1]):
    if BestFeaturesSelection[k][1]<=20:
        legend2.append(columns[k])
        plt.plot(BestFeaturesSelection[k])

plt.legend(legend2, loc='lower right')

plt.figure()
legend3=[]
for k in range(BestFeaturesSelection.shape[1]):
    if BestFeaturesSelection[k][2]<=20:
        legend3.append(columns[k])
        plt.plot(BestFeaturesSelection[k])

plt.legend(legend3, loc='lower right')

plt.figure()
legend4=[]
for k in range(BestFeaturesSelection.shape[1]):
    if BestFeaturesSelection[k][3]<=20:
        legend4.append(columns[k])
        plt.plot(BestFeaturesSelection[k])

plt.legend(legend4, loc='lower right')

plt.figure()
legend5=[]
for k in range(BestFeaturesSelection.shape[1]):
    if BestFeaturesSelection[k][4]<=20:
        legend5.append(columns[k])
        plt.plot(BestFeaturesSelection[k])

plt.legend(legend5, loc='lower right')

plt.show()
"""
