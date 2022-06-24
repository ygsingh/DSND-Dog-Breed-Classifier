from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm
import numpy as np
import cv2     
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.applications.xception import Xception


# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

# dog_names list
dog_names = ['ages/train/001.Affenpinscher',
             'ages/train/002.Afghan_hound',
             'ages/train/003.Airedale_terrier',
             'ages/train/004.Akita',
             'ages/train/005.Alaskan_malamute',
             'ages/train/006.American_eskimo_dog',
             'ages/train/007.American_foxhound',
             'ages/train/008.American_staffordshire_terrier',
             'ages/train/009.American_water_spaniel',
             'ages/train/010.Anatolian_shepherd_dog',
             'ages/train/011.Australian_cattle_dog',
             'ages/train/012.Australian_shepherd',
             'ages/train/013.Australian_terrier',
             'ages/train/014.Basenji',
             'ages/train/015.Basset_hound',
             'ages/train/016.Beagle',
             'ages/train/017.Bearded_collie',
             'ages/train/018.Beauceron', 
             'ages/train/019.Bedlington_terrier',
             'ages/train/020.Belgian_malinois',
             'ages/train/021.Belgian_sheepdog',
             'ages/train/022.Belgian_tervuren',
             'ages/train/023.Bernese_mountain_dog',
             'ages/train/024.Bichon_frise',
             'ages/train/025.Black_and_tan_coonhound',
             'ages/train/026.Black_russian_terrier',
             'ages/train/027.Bloodhound',
             'ages/train/028.Bluetick_coonhound',
             'ages/train/029.Border_collie',
             'ages/train/030.Border_terrier',
             'ages/train/031.Borzoi',
             'ages/train/032.Boston_terrier',
             'ages/train/033.Bouvier_des_flandres',
             'ages/train/034.Boxer',
             'ages/train/035.Boykin_spaniel',
             'ages/train/036.Briard',
             'ages/train/037.Brittany',
             'ages/train/038.Brussels_griffon',
             'ages/train/039.Bull_terrier',
             'ages/train/040.Bulldog',
             'ages/train/041.Bullmastiff',
             'ages/train/042.Cairn_terrier',
             'ages/train/043.Canaan_dog',
             'ages/train/044.Cane_corso',
             'ages/train/045.Cardigan_welsh_corgi',
             'ages/train/046.Cavalier_king_charles_spaniel',
             'ages/train/047.Chesapeake_bay_retriever',
             'ages/train/048.Chihuahua',
             'ages/train/049.Chinese_crested',
             'ages/train/050.Chinese_shar-pei',
             'ages/train/051.Chow_chow',
             'ages/train/052.Clumber_spaniel',
             'ages/train/053.Cocker_spaniel',
             'ages/train/054.Collie',
             'ages/train/055.Curly-coated_retriever',
             'ages/train/056.Dachshund',
             'ages/train/057.Dalmatian',
             'ages/train/058.Dandie_dinmont_terrier',
             'ages/train/059.Doberman_pinscher',
             'ages/train/060.Dogue_de_bordeaux',
             'ages/train/061.English_cocker_spaniel',
             'ages/train/062.English_setter',
             'ages/train/063.English_springer_spaniel',
             'ages/train/064.English_toy_spaniel',
             'ages/train/065.Entlebucher_mountain_dog',
             'ages/train/066.Field_spaniel',
             'ages/train/067.Finnish_spitz',
             'ages/train/068.Flat-coated_retriever',
             'ages/train/069.French_bulldog',
             'ages/train/070.German_pinscher',
             'ages/train/071.German_shepherd_dog',
             'ages/train/072.German_shorthaired_pointer',
             'ages/train/073.German_wirehaired_pointer',
             'ages/train/074.Giant_schnauzer',
             'ages/train/075.Glen_of_imaal_terrier',
             'ages/train/076.Golden_retriever',
             'ages/train/077.Gordon_setter',
             'ages/train/078.Great_dane',
             'ages/train/079.Great_pyrenees',
             'ages/train/080.Greater_swiss_mountain_dog',
             'ages/train/081.Greyhound',
             'ages/train/082.Havanese',
             'ages/train/083.Ibizan_hound',
             'ages/train/084.Icelandic_sheepdog',
             'ages/train/085.Irish_red_and_white_setter',
             'ages/train/086.Irish_setter',
             'ages/train/087.Irish_terrier',
             'ages/train/088.Irish_water_spaniel',
             'ages/train/089.Irish_wolfhound',
             'ages/train/090.Italian_greyhound',
             'ages/train/091.Japanese_chin',
             'ages/train/092.Keeshond',
             'ages/train/093.Kerry_blue_terrier',
             'ages/train/094.Komondor',
             'ages/train/095.Kuvasz',
             'ages/train/096.Labrador_retriever',
             'ages/train/097.Lakeland_terrier',
             'ages/train/098.Leonberger',
             'ages/train/099.Lhasa_apso',
             'ages/train/100.Lowchen',
             'ages/train/101.Maltese',
             'ages/train/102.Manchester_terrier',
             'ages/train/103.Mastiff',
             'ages/train/104.Miniature_schnauzer',
             'ages/train/105.Neapolitan_mastiff',
             'ages/train/106.Newfoundland',
             'ages/train/107.Norfolk_terrier',
             'ages/train/108.Norwegian_buhund',
             'ages/train/109.Norwegian_elkhound',
             'ages/train/110.Norwegian_lundehund',
             'ages/train/111.Norwich_terrier',
             'ages/train/112.Nova_scotia_duck_tolling_retriever',
             'ages/train/113.Old_english_sheepdog',
             'ages/train/114.Otterhound',
             'ages/train/115.Papillon',
             'ages/train/116.Parson_russell_terrier',
             'ages/train/117.Pekingese',
             'ages/train/118.Pembroke_welsh_corgi',
             'ages/train/119.Petit_basset_griffon_vendeen',
             'ages/train/120.Pharaoh_hound',
             'ages/train/121.Plott',
             'ages/train/122.Pointer',
             'ages/train/123.Pomeranian',
             'ages/train/124.Poodle',
             'ages/train/125.Portuguese_water_dog',
             'ages/train/126.Saint_bernard',
             'ages/train/127.Silky_terrier',
             'ages/train/128.Smooth_fox_terrier',
             'ages/train/129.Tibetan_mastiff',
             'ages/train/130.Welsh_springer_spaniel',
             'ages/train/131.Wirehaired_pointing_griffon',
             'ages/train/132.Xoloitzcuintli',
             'ages/train/133.Yorkshire_terrier']

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))



def predict_breed(img_path):
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Xception_model.add(Dense(133, activation='softmax'))
    Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')
    
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    Xception_prediction = Xception_model.predict(bottleneck_feature)
    
    return dog_names[np.argmax(Xception_prediction)]

def predict_image(img_path):
    
    if dog_detector(img_path):
        return "Dog {}".format(predict_breed(img_path).split('.')[1])
    if face_detector(img_path):
        return "Human who look like a {}".format(predict_breed(img_path).split('.')[1])
    else:
        return "Not a dog or human"
        