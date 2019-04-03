from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
import glob
import os

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, IMAGE_SIZE=96):
     # dataset parameters
    VALID_PATH = os.path.join(base_dir, 'train+val', 'valid')
    RESCALING_FACTOR = 1./255
      
    # instantiate data generators
    datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
    val_gen_full = datagen.flow_from_directory(VALID_PATH,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=16000,
                                             class_mode='binary',
                                             shuffle=False)
    return val_gen_full

val_gen_full = get_pcam_generators('C:\\Users\s167917\\Documents\\#School\\Jaar 3\\3 Project Imaging\\data') # path to data
x_val = val_gen_full[0][0]
y_val = val_gen_full[0][1]

# directory to the model and weights files (.json and .hdf5)
file_path = 'C:\\Users\\s167917\\Documents\\#School\\Jaar 3\\3 Project Imaging\\GitHub\\code\\dropout\\'

# get all file names in a list to loop over
file_names = glob.glob(file_path + '*.json')
model_names = []
auc_scores = []

for i in range(len(file_names)):
    file_name = os.path.splitext(file_names[i])[0]
    print(os.path.split(file_name)[1],end = '\t')
    
    # get model architecture and weights
    json_file = open(file_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(file_name+'_weights.hdf5')
    
    # calculate and print auc score
    predictions = loaded_model.predict([x_val])
    model_names.append(os.path.split(file_name)[1])
    auc_scores.append(roc_auc_score(y_val,predictions))
    print(auc_scores[-1])
    
    # clear memory
    backend.clear_session()
    
    # write to excel file (can be moved out of the loop, this prevents losing data when something goes wrong)
    data_frame = DataFrame({'model': model_names, 'AUC score': auc_scores})
    data_frame.to_excel(file_path+'auc_scores.xlsx', sheet_name='sheet1', index=False)