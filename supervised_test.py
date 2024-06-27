from utils import get_far_mdr, time_to_detection, get_test_files
import pandas as pd
import numpy as np
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
import gc

far_dict = {'mean':np.zeros(20), 'std':np.zeros(20)}
mdr_dict = {'mean':np.zeros(20), 'std':np.zeros(20)}
ttd_dict = {'mean':np.zeros(20), 'std':np.zeros(20)}

arch = 1

idvs = [idv for idv in range(1,21)]

for idv in idvs: 

    print(f'========== IDV {idv} ==========')
    x_test, y_test = get_test_files(idv = idv, n_test = 1000, w = 5, standard = True)

    model_path = f'supervised_learning/models/idv{idv}/arch{arch}/best_model'
    model = load_model(model_path)

    x_test_reshape = []
    for x in x_test:
        x = x.reshape(x.shape[0], -1)
        x_test_reshape.append(x) 

    far_list = []
    mdr_list = []
    ttd_list = []

    y_test = y_test[0]

    for i, x in enumerate(x_test_reshape):

        if (i+1) % 100 == 0:
            print(f'------ #{i+1} ------')
        
        far, mdr = get_far_mdr(model, x, y_test)    
        y_pred = model.predict(x)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        ttd = time_to_detection(y_pred)
        far_list.append(far)
        mdr_list.append(mdr)
        ttd_list.append(ttd)

    far = np.mean(far_list)
    far_std = np.std(far_list)
    mdr = np.mean(mdr_list)
    mdr_std = np.std(mdr_list)
    ttd = np.mean(ttd_list)
    ttd_std = np.std(ttd_list)

    print(f'IDV {idv}:')
    print('-----------------')
    print(f'FAR: {far*100:.2f}%')
    print(f'MDR: {mdr*100:.2f}%')
    print(f'TTD: {ttd:.2f} time units')
    print('-----------------')

    far_dict['mean'][idv-1] = far
    far_dict['std'][idv-1] = far_std

    mdr_dict['mean'][idv-1] = mdr
    mdr_dict['std'][idv-1] = mdr_std

    ttd_dict['mean'][idv-1] = ttd
    ttd_dict['std'][idv-1] = ttd_std

    dict = {'IDV':np.arange(1,21,1),
            'FAR_mean':far_dict['mean'],
            'FAR_std':far_dict['std'],
            'MDR_mean':mdr_dict['mean'],
            'MDR_std':mdr_dict['std'],
            'TTD_mean':ttd_dict['mean'],
            'TTD_std':ttd_dict['std'],
            'arch':arch*np.ones(20)}

    import pandas as pd
    df = pd.DataFrame(dict)
    print(df)

    clear_session()
    gc.collect()

    df.to_excel(f'supervised_results_arch{arch}.xlsx')

    