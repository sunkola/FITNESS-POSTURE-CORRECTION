import tensorflow as tf
from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping 
from keras import layers, optimizers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def get_filename_number(s):#得到mp4檔名稱裡面的數字，這樣才能做排序
    return int(s.split('_')[-1].split('.')[0])
def get_filename_number_1(s):
    return int(s)

def get_csvdata(csv_file_path):
    csv_path = []
    dir_files = os.listdir(csv_file_path)
    sorted_dir = sorted(dir_files,key=get_filename_number_1)
    for dirs in sorted_dir:
        # file_path = f'./csv_dataset/{dirs}/'
        file_path = os.path.join(csv_file_path,dirs)
        files =os.listdir(file_path)
        sorted_files = sorted(files,key=get_filename_number)
        for items in sorted_files:
            # file_vedio_path = f'{file_path}{items}'
            file_vedio_path = os.path.join(file_path,items)
            csv_path.append((dirs,file_vedio_path))
        dir_label_df = pd.DataFrame(csv_path,columns=['label','csv_path'])
        dir_label_df['label'] = dir_label_df['label'].apply(lambda x:int(x))
    return dir_label_df

def load_csv_from_csvpath(dir_label_df):
    """_summary_

    Args:
        dir_label_df (_type_): _description_

    Returns:
        _type_: tuple
    """
    num_samples = 10
    frame_length = 50
    X = []
    Y = []
    for item_label,item_csvpath in dir_label_df.values:
        data = pd.read_csv(item_csvpath)
        data_label = item_label
        
        if len(data)<frame_length:
            continue
        else:
            
            data_1 = data.iloc[:int(len(data)*0.3)] #data的前30%
            selected_indices_1 = np.random.choice(data_1.index,size=num_samples,replace=False)
            selected_data_1 = data_1.loc[selected_indices_1]
            sorted_data_1 = selected_data_1.sort_index()
            
            data_2 = data.iloc[int(len(data)*0.3)+1:int(len(data)*0.7)]
            selected_indices_2 = np.random.choice(data_2.index,size=num_samples,replace=False)
            selected_data_2 = data_2.loc[selected_indices_2]
            sorted_data_2 = selected_data_2.sort_index()
        
            data_3 = data.iloc[-int(len(data)*0.3):]
            selected_indices_3 = np.random.choice(data_3.index,size=num_samples,replace=False)
            selected_data_3 = data_3.loc[selected_indices_3]
            sorted_data_3 = selected_data_3.sort_index()
            
            data_final = pd.concat([sorted_data_1,sorted_data_2,sorted_data_3],ignore_index=True)
        
            
        #轉換為numpy array 輸出
        numpy_data = data_final.to_numpy()
        numpy_label_data = np.int64(data_label)
        X.append(numpy_data)
        Y.append(numpy_label_data)
        ndarray_data = np.array(X) #ndarray_data (3,30,99) ##3個csv,30個row,99個欄位
        ndarray_label_data = np.array(Y)
        # ndarray_label_data = ndarray_label_data.reshape(len(Y),1)
        encoder = OneHotEncoder(categories=[range(13)], sparse=False)
        ndarray_label_data_onehot = encoder.fit_transform(ndarray_label_data.reshape(-1, 1))
    return ndarray_label_data_onehot,ndarray_data


def create_model():
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(30, 99)))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(13, activation='softmax'))
    adam = optimizers.Adam(learning_rate=1e-3) 
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy() , optimizer= adam, metrics=['categorical_accuracy'])
    return model

def train(train_X,train_y,batch_size,epochs):
    model = create_model()
    model.summary()
    model_mckp = ModelCheckpoint('best_model_weights.h5', 
                                  monitor='val_categorical_accuracy', 
                                  save_best_only=True,      
                                  save_weights_only=True,  
                                  mode='max')
    
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy',
                                   patience=20,
                                   verbose=0,
                                   restore_best_weights=True)
    
    history = model.fit(train_X, train_y,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.25,
                  callbacks=[model_mckp, early_stopping])
    return model, history

def load(weight='best_model_weights.h5'):
    model = create_model()
    model.load_weights(weight)
    return model

    
if __name__ == "__main__":
    csv_file_path = 'csv_dataset_1'
    dir_label_df= get_csvdata(csv_file_path)
    train_data , test_data =train_test_split(dir_label_df,random_state=777,train_size=0.8)
    train_X = load_csv_from_csvpath(train_data)[1]
    test_X =  load_csv_from_csvpath(test_data)[1]
    train_y = load_csv_from_csvpath(train_data)[0]
    test_y = load_csv_from_csvpath(test_data)[0]
    
    #hyperparameters
    epochs = 350
    batch_size = 60
    n_timesteps, n_features, n_outputs =  train_X.shape[1], train_X.shape[2], train_y.shape[1]#train_X.shape(, 30, 99)#train_y.shape(517, 22)
    
    model, history = train(train_X=train_X,train_y=train_y,batch_size=batch_size,epochs=epochs)
    y_pred = model.predict(test_X)
    y_pred_classes = []
    for item in y_pred:
        max_index = np.argmax(item)
        new_array = np.zeros_like(item)
        new_array[max_index] = 1
        y_pred_classes.append(new_array)
    y_pred_classes = np.array(y_pred_classes)
    cm = confusion_matrix(test_y.argmax(axis=1), y_pred_classes.argmax(axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
     
    accuracy =  model.evaluate(test_X, test_y, batch_size=len(test_X), verbose=1) 