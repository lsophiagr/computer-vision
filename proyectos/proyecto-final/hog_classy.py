#Sophia Gamarro 

def get_histogram_values_and_positions(angle, magnitud):

  bin_of_angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]

  if angle > 180:
    angle = angle-180

  bin_position = bisect_left(bin_of_angles, angle+1)
  bin_position = bin_position - 1 # -1 porque el index empieza en 0

  # ==Ejemplo==
  # angulo = 36
  # bin_position = 2 -> esta en la posicion 2 de los rangos del bin
  # start_angle = 20
  # end_angle = 40


  start_angle = bin_of_angles[bin_position] 
  end_angle = start_angle + 20

  

  substr_bin_1 = end_angle - angle
  substr_bin_2 = angle - start_angle
  

  bin_magnitud = ( substr_bin_1 / 20 ) * magnitud
  bin_magnitud_2 = ( substr_bin_2 / 20 ) * magnitud



  bin_position_2 = bin_position + 1

  if bin_position == 9:
    bin_position_2 = 0 #da la vuelta al array para tomar el bin inicial

  return bin_magnitud, bin_magnitud_2, bin_position , bin_position_2


def create_patch_histogram(patch, mag_patch, ang_patch):

  patch_histogram = np.zeros(9, dtype='float32')

  for x in range(0, 8): #filas
    for y in range(0, 8): #columnas

      current_mag = mag_patch[x, y]
      current_angle = ang_patch[x, y]

      mag_1, mag_2, pos_1, pos_2 = get_histogram_values_and_positions(current_mag, current_angle)

      patch_histogram[pos_1] = patch_histogram[pos_1] + mag_1
      patch_histogram[pos_2] = patch_histogram[pos_2] + mag_2


  return patch_histogram


def create_histograms_per_patch(img, magnitudes, angles):

  histograms = np.zeros((16, 8, 9), dtype='float32') #almacenaremos los histogramas (length 9) en una matriz de 16x8 por los patches de 8x8
  fila = -1
  columna = -1

  for x in range(0, img.shape[0], 8): #recorre las filas

    histogram_patch = []
    columna = -1

    for y in range (0, img.shape[1], 8): #recorre las columnas

      patch = img[x:x+8, y:y+8]
      mag_patch = magnitudes[x:x+8, y:y+8]
      ang_patch = angles[x:x+8, y:y+8]

      histogram_patch = create_patch_histogram(patch, mag_patch, ang_patch)
      columna += 1

    fila += 1

    histograms[fila, columna] = histogram_patch
  
  return histograms


def normalise_histograms(whole_img_histograms):

  normalised_histograms = np.zeros((15, 7, 36), dtype='float32') #almacenaremos los histogramas normalizados (length 36)
  fila = -1
  columna = -1                                                # en una matriz de 15x7

  for x in range(0, 15): #filas

    normalised_vector= []
    columna = -1

    for y in range(0, 7): #columnas

      hist_patch = whole_img_histograms[x: x+2, y: y+2] #tomamos los histogramas en en cuadrados de 2X2 para normalizar
      hist_flatten = hist_patch.flatten() # para crear un array de 36x1
      k_value = np.sqrt(np.sum(np.square(hist_flatten), axis=0))
      
      if k_value == 0:
        normalised_vector = hist_flatten
      else:
        normalised_vector = hist_flatten / k_value


      columna += 1

    fila += 1

    normalised_histograms[fila, columna] = normalised_vector

  return normalised_histograms


def get_hog_features(img):

  # Calcular gradient
  gradientX = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
  gradientY = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

  # Python Calcula el gradient magnitud y  direccion ( in grados )
  magnitudes, angles = cv2.cartToPolar(gradientX, gradientY, angleInDegrees=True)

  whole_img_histograms = create_histograms_per_patch(img, magnitudes, angles)

  whole_normalised_hist = normalise_histograms(whole_img_histograms)

  final_features = whole_normalised_hist.flatten() # 378
  
  return final_features

def img_pyramids(image, scale=1.5, minSize=(64, 128)):
  # yield la imagen original
  yield image

  # loop para el resize
  while True:
    #rescale
    w = int(image.shape[1] / scale) #width 
    image = imutils.resize(image, width=w)

    # si la imagen no tiene el tamaño minimo detiene el loop
    if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
      break
    
    yield image


def sliding_window(image, stepSize, windowSize):
  
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):

      # retorna la window actual
      yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def nms_score(candidates, overlap_threshold=0.5,debug=True):
  if len(candidates) < 2:
      return candidates
  
  if candidates.dtype.kind == "i":
    candidates = candidates.astype("float")

  non_supressed_boxes = []
  """
  p1. x1,y1 -> x,y
    * -----
      |     |
      |     |
      ----- * p2. x2,y2 -> x+w,y+h
  """
  
  
  # grab the coordinates of the bounding boxes
  x1 = candidates[:,0]
  y1 = candidates[:,1]
  x2 = candidates[:,0]+candidates[:,2]
  y2 = candidates[:,1]+candidates[:,3]

  # we extract the confidence scores as well
  scores = candidates[:, 4]

 
  #compute the area of the bounding boxes w*h
  box_areas = candidates[:,2]*candidates[:,3]
  #idxs = np.argsort(y2)

  #sort the prediction boxes in P
  #according to their confidence scores
  idxs = scores.argsort()

  
  selected = []

  while len(idxs) > 0:
      # grab the last index in the indexes list and add the
      # index value to the list of picked indexes
      last = len(idxs) - 1
      last_idx = idxs[last]
      #print(f'last_idx: {last_idx}')
      selected.append(last_idx)
      
      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = np.maximum(x1[last_idx], x1[idxs[:last]])
      yy1 = np.maximum(y1[last_idx], y1[idxs[:last]])
      xx2 = np.minimum(x2[last_idx], x2[idxs[:last]])
      yy2 = np.minimum(y2[last_idx], y2[idxs[:last]])

      

      # compute the width and height of the bounding box
      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)    

      #compute the ratio of overlap
      overlap = (w * h) / box_areas[idxs[:last]]
      
      if debug:
          print(f'overlap: {overlap}')

      max_score_pos = np.argmax(candidates[idxs[np.concatenate(([last],np.where(overlap > overlap_threshold)[0]))],4])
      
      if debug:
          print(f'SCR:{candidates[idxs[max_score_pos]]}')
      #selected.append(idxs[max_score_pos])
      
      # delete all indexes from the index list that have been tested
      idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlap_threshold)[0])))

      
  return candidates[selected].astype("int")



def detect_pedestrians(image, winW, winH):

  pyramid_count = 0
  candidates = np.empty((0,6), dtype='float32')
  resize_info = []

  minSizeY = image.shape[0] * 0.2
  minSizeX = image.shape[1] * 0.2

  # Obtenemos cada piramide de la imagen inicial
  for resizedImg in img_pyramids(image, scale=1.5, minSize=[minSizeX, minSizeY]):

    pyramid_count += 1

    # aplicamos la sliding window a cada layer de la piramide
    for (x, y, window) in sliding_window(resizedImg, stepSize=128, windowSize=[winW, winH]):

      # Si la window restante es de tamaño diferente a la que queremos , salta a la siguiente piramide.
      # Esto pasa porque a veces ya recorrimos la mayor parte de la piramide y solo quedan unos cuantos pixeles
      # faltantes por lo tanto no se ajusta al tamaño y/o aspect-ratio de la window
      if window.shape[0] != winH or window.shape[1] != winW:
        continue
      

      # CLASSIFIER
      img = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
      img = cv2.resize(img, (64, 128))
      img = np.float32(img) / 255.0

      hog = np.zeros(378, dtype='float32')
      hog = get_hog_features(img)

      prediction = svm_model.predict(hog.reshape(1, -1))[0]
      confidence_score = svm_model.decision_function(hog.reshape(1, -1))[0]

      if prediction == 'Pedestrians':
        candidates = np.append(candidates, np.array([[x, y, x+64, y+128, confidence_score, pyramid_count]], dtype='float32'), axis=0)


  final_candidates = nms_score(candidates, overlap_threshold=0.5, debug=False)


  for i in final_candidates:
    #info para luego dibujar las windows detectadas en los puntos y escala correcta
    resize_info.append({'pts': (i[0], i[1]), 'pyramid_n' : i[5]}) #punto en X, punto en Y, en que piramide se encontró 


  return resize_info
    

def draw_rectangles(image, resize_info, windowW, windowH):

  colors = [(0, 255, 0), (38, 235, 208), (69, 72, 230), (163, 235, 40)] 
  indx = 0
  clone = image.copy()

  for info in resize_info:

    #coordenada en X-Y * piramide numero N
    start_point = info['pts'][0] * info['pyramid_n'] , info['pts'][1] * info['pyramid_n'] 
    windowW = windowW * info['pyramid_n']
    windowH = windowH * info['pyramid_n']

    end_point = start_point[0] + windowW, start_point[1] + windowH
    color = (0, 255, 0)#green

    cv2.rectangle(clone, start_point, end_point, colors[indx] , 4)

    indx += 1
    if indx >= len(colors) :
      indx = 0
    
    
  return clone
  


if __name__ == '__main__':
  
  import sys
  import cv2
  import pickle
  import imutils
  import numpy as np
  from bisect import bisect_left
  from sklearn.svm import LinearSVC
  
  
  # load the model from disk
  #filename = 'svm_trained_model3.sav'
  #svm_model = pickle.load(open(filename, 'rb'))

  filename = 'svm_trained_model.pickle'
  with open(filename, 'rb') as f:
    svm_model = pickle.load(f)

  path = sys.argv[1]
  image = cv2.imread(path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  
  winW = 250 #tamaño de window que recorrerá toda la imagen
  winH = winW * 2 #aspect-ratio de 1:2 (WxH)

  resize_info = detect_pedestrians(image, winW, winH)
  final_image = draw_rectangles(image, resize_info, winW, winH)

  cv2.imwrite('./{}_result.jpg'.format(path), cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
  

