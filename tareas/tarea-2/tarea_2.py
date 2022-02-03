#Sophia Gamarro

import cv2 as cv

#camaras
cap = cv.VideoCapture(0)

#Version de camara
(major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')

if int(major_ver)  < 3 :
    fps = cap.get(cv.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {}".format(fps))
else :
    fps = cap.get(cv.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {}".format(fps))



window_name = 'Original'
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
window_name2 = 'Pencil Sketch Effect'
cv.namedWindow(window_name2, cv.WINDOW_NORMAL)


# Captura frame-by-frame
ret, frame = cap.read()

#arrange de las ventanas
height_frame, width_frame = frame.shape[0:2] #dimensiones de la captura
k = 2
height = int(height_frame/k)
width = int(width_frame/k)


#align de las ventanas
cv.moveWindow(window_name, 0,0)
cv.moveWindow(window_name2, width, 0) #+width ventana anterior



def pencil_sketch_grey(img):
    #funcion nativo de opencv 
    sk_gray, sk_color = cv.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1) 
    return  sk_gray

    

while(True):

    # Captura frame-by-frame
    ret, frame = cap.read()

    #aplicacion del filtro
    result = pencil_sketch_grey(frame)

    cv.resizeWindow(window_name, (int(width/2),int(height/2)))
    cv.resizeWindow(window_name2, (width,height))


    #imagen original
    cv.imshow(window_name,frame)

    #filtro aplicado
    cv.imshow(window_name2 ,result)# result = imagen ya procesada



    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# Release de la captura de camara
cap.release()
cv.destroyAllWindows()
