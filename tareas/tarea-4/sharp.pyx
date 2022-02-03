import numpy as np
import cv2 as cv

cimport numpy as np

np.import_array()  # needed to initialize numpy-API

cdef bint ret
cdef int padding
cdef np.ndarray padded_img
cdef np.ndarray new_img
cdef np.ndarray block_3x3 #porcion de la imagen para realizar la convolusion

#dimensiones de la imagen final
cdef int height, cwidth

#indexes para la convolusion
cdef int row, col

#cpdef np.ndarray[int, ndim=2] add_padding([int, ndim=2] img, int r):
cpdef np.ndarray add_padding(np.ndarray img, int r):
    #img: array de 2 dimensiones que representa una imagen en escala de grises
    #r: numero de pixeles que tendra cada lado como padding

    #cdef Py_ssize_t height = img.shape[0] + (r*2)
    #cdef Py_ssize_t width = img.shape[1] + (r*2)

    cdef int height = img.shape[0] + (r*2)
    cdef int width = img.shape[1] + (r*2)


    return np.full( (height, width), 255) 
    #return np.zeros(shape=(height, width), dtype=np.int32)

    #cdef cnp.npy_intp dim = 0
    #print( np.PyArray_SimpleNew(height, &width, np.NPY_INT32) )


cpdef np.ndarray reassign_img(np.ndarray padded_img, np.ndarray img, int r):
    #padded_img: image en blanco con padding
    #img: imagen original
    #r: padding cada en lado
    
    padded_img[r:-r , r:-r] =  img
    return padded_img


cpdef int convolution(np.ndarray sub_img, np.ndarray kernel):
    #sub_img: sub-imagen de 3x3
    #kernel: kernel a aplicar de 3x3
    return np.sum( (sub_img * kernel) )


cpdef np.ndarray sharp_filter(np.ndarray img, np.ndarray kernel):
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


cdef str window_name2 = 'Sharp Effect'
cv.namedWindow(window_name2, cv.WINDOW_NORMAL)

# Captura frame-by-frame
#cpdef [int, ndim=3] frame 
ret, frame = cap.read()

 #dimensiones de la captura
cdef int height_frame, width_frame
height_frame, width_frame = frame.shape[0:2]
cdef int k = 2

#tamaño de la ventana para visualizar
height_frame = height_frame/k
width_frame = width_frame/k

cv.resizeWindow(window_name2, (width_frame,height_frame)) #sea la mitad del tamaño
cv.moveWindow(window_name2, width_frame, 0) #mover la ventana


#kernel
cdef np.ndarray sharpen_kernel = np.array( [[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])

cdef int cont = 1
cdef np.ndarray B, G, R


while(True):

    # Captura frame-by-frame
    ret, frame = cap.read()

    B, G, R = cv.split(frame)

    #aplicacion del filtro
    R = sharp_filter(R, sharpen_kernel)
    G = sharp_filter(G, sharpen_kernel)
    B = sharp_filter(B, sharpen_kernel)

    #filtro aplicado
    cv.imshow(window_name2 , cv.convertScaleAbs(cv.merge([B, G, R])) )#imagen ya procesada


    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# Release de la captura de camara
cap.release()
cv.destroyAllWindows()