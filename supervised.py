from supervised_learning.model import get_model, train_batch
from utils import get_files, get_test_files
import numpy as np
import gc
from tensorflow.keras.backend import clear_session

clear_session()
gc.collect()

arch = 1
epochs = 3000


idv_list = [x for x in range(1,21)]
idv_list = [18]

for idv in idv_list:

    print(f'=============== IDV {idv} ===============')

    save_path = f"supervised_learning/models/idv{idv}/arch{arch}/test_reg"
    model_name = f"/idv{idv}_arch{arch}"
    
    x_train, y_train, x_val, y_val, max_values, min_values = get_files(
    idv=idv, n_train=3000, n_val=600, w=5, standard=True
)
    
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)


    model = get_model(arch=arch, l2_reg = 0.01, learning_rate = 0.005, optimizer = 'adam')
    train_batch(model=model, x=x_train, y=y_train, x_val=x_val, y_val=y_val, epochs=epochs, best_model_path = save_path + '/best_model')

    model.save(save_path + model_name)

    clear_session()
    del x_train, y_train, x_val, y_val, max_values, min_values, model, x_test, y_test, x_test_reshape
    gc.collect()
