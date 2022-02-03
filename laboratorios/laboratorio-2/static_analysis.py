import cv2 as cv
import numpy as np

def pyramid(img, scale=0.6, min_size=(8,8)):
    """ Build a pyramid for an image until min_size
        dimensions are reached.
    Args: 
        img (numpy array): Source image
        scale (float): Scaling factor
        min_size (tuple): size of pyramid top level.
    Returns:
        Pyramid generator
    """
    yield img
    
    while True:
        img = cv.resize(img, None,fx=scale, fy=scale, interpolation = cv.INTER_CUBIC)
        if ((img.shape[0]<min_size[0]) and img.shape[1]<min_size[1]):
            break
        yield img

def search_logo_ccoeff_normed(imgToSearchColor, imgToSearchGray, imgTemplate):
    w, h = imgTemplate.shape[::-1]
    print('Size of img Template ', imgTemplate.shape)
    res = cv.matchTemplate(imgToSearchGray, imgTemplate, cv.TM_CCOEFF_NORMED)
    # threshold = 0.50
    # loc = np.where(res >= threshold)
    # for pt in zip(*loc[::-1]):
    #     print('puntos mayores a ', pt)
    #     cv.rectangle(imgToSearchColor, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    print('TM_CCOEFF_NORMED ', top_left, bottom_right)
    cv.rectangle(imgToSearchColor, top_left, bottom_right, (0,255,255), 2)


    # threshold = 0.5
    # if (max_val >= threshold):
    #     cv.rectangle(imgToSearchColor, top_left, bottom_right, (0,255,255), 2)
    
    return imgToSearchColor

def search_logo_sqdiff_normed(imgToSearchColor, imgToSearchGray, imgTemplate):
    w, h = imgTemplate.shape[::-1]
    # print('Size of img Template ', imgTemplate.shape)
     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    res = cv.matchTemplate(imgToSearchGray, imgTemplate, cv.TM_SQDIFF_NORMED)

    print('res MAX', np.amax(res))
    print('res MIN', np.amin(res))
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    print('sss ', min_loc)
    print('valueee ', min_val)
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv.rectangle(imgToSearchColor, top_left, bottom_right, (0,255,255), 2)
    
    # threshold = 0.8
    # if (min_val >= threshold):
    #     # print('TM_CCOEFF_NORMED ', top_left, bottom_right)
    #     cv.rectangle(imgToSearchColor, top_left, bottom_right, (0,255,255), 2)

    return imgToSearchColor

if __name__ == "__main__":
    # Read color image
    imgTestedColor = cv.imread('./imgs/tests/test2.png')
    # Converting the img to grey
    # imgTestedGray = cv.cvtColor(imgTestedColor, cv.COLOR_BGR2GRAY)

    # Read Gary color image
    imgTestedGray = cv.imread('./imgs/tests/test2.png')
    imgTestedGray = cv.cvtColor(imgTestedColor, cv.COLOR_BGR2GRAY)

    # templateImg = cv.imread('./imgs/templates/Logo-UFM.png', cv.IMREAD_GRAYSCALE)
    templateImg = cv.imread('./imgs/templates/logo0.png')
    templateImg = cv.cvtColor(templateImg, cv.COLOR_BGR2GRAY)

    print('size of img ', templateImg.shape)
    w, h = templateImg.shape[::-1]
    print('taammaa ', w, ' alturaa ', h)
    for temp in pyramid(templateImg):
        # Check that template size is not bigger than the image in what we are looking in.
        if (temp.shape[0]>imgTestedGray.shape[0] or temp.shape[1]>imgTestedGray.shape[1]):
            print('The template is bigger than the image')
        else:
                
            # a = search_logo_ccoeff_normed(imgTestedColor, imgTestedGray, temp)
            b = search_logo_ccoeff_normed(imgTestedColor, imgTestedGray, temp)
    
    cv.imshow('detected ', b)
    cv.waitKey(0)
    cv.destroyAllWindows()



    # res = cv.matchTemplate(img_gray, templateImg, cv.TM_CCOEFF_NORMED)
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)

    # print('TM_CCOEFF_NORMED ', top_left, bottom_right)
    # cv.rectangle(img_tested, top_left, bottom_right, (0,255,255), 2)


    # threshold = 0.50
    # loc = np.where(res >= threshold)
    # print('holaa ', loc)
    # for pt in zip(*loc[::-1]):
    #     print('puntos mayores a ', pt)
    # #     cv.rectangle(img_tested, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)


    # res = cv.matchTemplate(img_gray, templateImg, cv.TM_CCOEFF)
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)

    # print('TM_CCOEFF ', top_left, bottom_right)
    # cv.rectangle(img_tested, top_left, bottom_right, (0,255,0), 2)


    # res = cv.matchTemplate(img_gray, templateImg, cv.TM_CCORR)
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)

    # print('TM_CCORR ', top_left, bottom_right)
    # cv.rectangle(img_tested, top_left, bottom_right, (255,0,0), 2)

    # res = cv.matchTemplate(img_gray, templateImg, cv.TM_CCORR_NORMED)
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)

    # print('TM_CCORR_NORMED ', top_left, bottom_right)
    # cv.rectangle(img_tested, top_left, bottom_right, (255,0,255), 2)

    #  # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # res = cv.matchTemplate(img_gray, templateImg, cv.TM_SQDIFF)
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # top_left = min_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)

    # print('TM_CCORR_NORMED ', top_left, bottom_right)
    # cv.rectangle(img_tested, top_left, bottom_right, (100,0,100), 5)

    #  # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # res = cv.matchTemplate(img_gray, templateImg, cv.TM_SQDIFF_NORMED)
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # top_left = min_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)

    # print('TM_SQDIFF_NORMED ', top_left, bottom_right)
    # cv.rectangle(img_tested, top_left, bottom_right, (0,200,100), 5)

    # cv.imshow('detected ', img_tested)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

