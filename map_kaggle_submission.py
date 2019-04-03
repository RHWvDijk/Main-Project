import os
import numpy as np
import glob
import pandas as pd
from matplotlib.pyplot import imread
from keras.models import model_from_json
from keras import backend

# directory to test data
test_path = 'C:\\Users\\s167917\\Documents\\#School\\Jaar 3\\3 Project Imaging\\data\\test\\'

# directory to the model and weights files (.json and .hdf5)
file_path = 'C:\\Users\\s167917\\Documents\\#School\\Jaar 3\\3 Project Imaging\\GitHub\\code\\blocks\\'

# make list of all models
file_names = glob.glob(file_path + '*.json')
for i in range(len(file_names)):
    file_names[i] = os.path.splitext(file_names[i])[0]

# open the test set in batches (as it is a very big dataset) and make predictions
test_files = glob.glob(test_path + '*.tif')
file_batch = 5000
max_idx = len(test_files)

for file_name in file_names:
    print("\n"+os.path.split(file_name)[1])
    # load model and model weights
    json_file = open(file_name+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(file_name+"_weights.hdf5")
    submission = pd.DataFrame()    
    for idx in range(0, max_idx, file_batch):
        print('Indexes: %i - %i'%(idx, min(idx+file_batch,max_idx)))
        test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})

        # get the image id 
        test_df['id'] = test_df.path.map(lambda x: x.split(os.sep)[-1].split('.')[0])
        test_df['image'] = test_df['path'].map(imread)
        K_test = np.stack(test_df['image'].values)

        # apply the same preprocessing as during draining
        K_test = K_test.astype('float')/255.0
        predictions = model.predict(K_test)
        test_df['label'] = predictions
        submission = pd.concat([submission, test_df[['id', 'label']]])

    # save your submission
    submission.head()
    submission.to_csv(file_name+".csv", index = False, header = True)
    backend.clear_session()
