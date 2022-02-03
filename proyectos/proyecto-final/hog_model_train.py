#Sophia Gamarro 

import os
import cv2
#import argparse
#import pickle
from sklearn.svm import LinearSVC
from hog_classifier import get_hog_features


images = []
labels = []

image_folders = ['Background', 'Pedestrians']
#image_paths = ['./Pedestrians-Dataset/']
main_folder = './Pedestrians-Dataset'

for sub_folder in image_folders:
    
  # obtener el nombre de todas la imagenes
  all_images = os.listdir(f"{main_folder}/{sub_folder}")
  
  # recorrer todas las imagenes
  for image in all_images:
    image_path = f"./{main_folder}/{sub_folder}/{image}"

    # Leer imagen
    img = cv2.imread(image_path,0)
    #img = cv2.resize(img, (64, 128))
    img = np.float32(img) / 255.0


    hog_desc = np.zeros(378, dtype='float32')
    hog_desc = get_hog_features(img)
    

    # update la data y labels
    images.append(hog_desc)
    labels.append(sub_folder)

# train Linear SVC 
print('Entrenando en imagenes...')
svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(images, labels)
print('Entrenamiento finalizado')

#guardar modelo entrenado en disco
#filename = 'svm_trained_model3.sav'
#pickle.dump(svm_model, open(filename, 'wb'))
