#Sophia Gamarro

import cv2 as cv
import numpy as np

def add_padding(img, r):

    #img: array de 2 dimensiones que representa una imagen en escala de grises
    #r: numero de pixeles que tendra cada lado como padding

    return np.full( (img.shape[0]+r*2, img.shape[1]+r*2), 255)


def reassign_img(padded_img, img, r):
    #padded_img: image en blanco con padding
    #img: imagen original
    #r: padding cada en lado
    
    padded_img[r:-r , r:-r] =  img
    return


def convolution(sub_img, kernel):
    #sub_img: sub-imagen de 3x3
    #kernel: kernel a aplicar de 3x3
    return np.sum( (sub_img * kernel) )


def sharp_filter(img, kernel):
    #aplicar padding
    padding = 1

    padded_img = add_padding(img, padding)
    reassign_img(padded_img, img, padding)

    new_img = add_padding(img, padding)

    #dimensiones
    height = padded_img.shape[0]
    width = padded_img.shape[1]


    for row in range(1, height - (padding*2)): 
        for col in range(1, width - (padding*2)):
            
            block_3x3 = padded_img[row : row+3, col : col+3] #sub-imagen de 3x3
            new_img[row][col] = convolution(block_3x3, kernel)


    return new_img   



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



window_name2 = 'Sharp Effect'
cv.namedWindow(window_name2, cv.WINDOW_NORMAL)

# Captura frame-by-frame
ret, frame = cap.read()

#arrange de las ventanas
height_frame, width_frame = frame.shape[0:2] #dimensiones de la captura
k = 2
height = int(height_frame/k)
width = int(width_frame/k)
cv.moveWindow(window_name2, width, 0) #+width ventana anterior



#kernel
sharpen_kernel = np.array( [[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])


while(True):

    # Captura frame-by-frame
    ret, frame = cap.read()
    (B, G, R) = cv.split(frame)

    #aplicacion del filtro
    R = sharp_filter(R, sharpen_kernel)
    G = sharp_filter(G, sharpen_kernel)
    B = sharp_filter(B, sharpen_kernel)

    cv.resizeWindow(window_name2, (width,height))

    #filtro aplicado
    cv.imshow(window_name2 , cv.convertScaleAbs(cv.merge([B, G, R])) )#imagen ya procesada


    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# Release de la captura de camara
cap.release()
cv.destroyAllWindows()


'''


img = cv.imread('./lenna.jpg')
(B, G, R) = cv.split(img)

#kernel
sharpen_kernel = np.array( [[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

R = sharp_filter(R, sharpen_kernel)
G = sharp_filter(G, sharpen_kernel)
B = sharp_filter(B, sharpen_kernel)

cv.imwrite('./test_lenna.jpg', cv.convertScaleAbs(cv.merge([B, G, R])))

'''




