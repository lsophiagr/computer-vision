# Manuel Alexander Palencia Gutierrez
# Sophia Gamarro 
from video import CountsPerSec, VideoCaptureThread, ImShowThread
import argparse
import cv2 as cv
import numpy as np
from threading import Thread

def pyramid(img, scale=0.3, min_size=(50,50)):
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
    """
    Search a template in img array using template matching, multiple objs detection using threshold.
    """
    w, h = imgTemplate.shape[::-1]
    # print('Size of img Template ', imgTemplate.shape)
    res = cv.matchTemplate(imgToSearchGray, imgTemplate, cv.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        print('Points where obj is detected ', pt)
        cv.rectangle(imgToSearchColor, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)
    return imgToSearchColor

def img_annotate(img, text):
    """ Annotate an image with text
    """
    cv.putText(img, text,(10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0))
    return img

def windowThread(source=0):
    """
    Dedicated thread for showing video frames with VideoShow object.
    Main thread grabs video frames.
    """
    cap = cv.VideoCapture(source)
    ret, frame = cap.read()
    win_title = 'THREADED WINDOW'
    winThreaded = ImShowThread(frame, win_title).start()
    cps = CountsPerSec().start()  

    # templateImg = cv.imread('./imgs/templates/Logo-UFM.png', cv.IMREAD_GRAYSCALE)
    templateImg = cv.imread('./imgs/templates/logo2.png')
    templateImg = cv.cvtColor(templateImg, cv.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        imgTestedGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if not ret or winThreaded.stopped:
            winThreaded.stop()
            break
        fps = str(round(cps.freq(),2))
        # frame = img_annotate(frame, fps)
        
        for temp in pyramid(templateImg):
            b = search_logo_ccoeff_normed(frame, imgTestedGray, temp)
        winThreaded.frame = b
        cps.increment()

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

class VideoCaptureThread:
    """ Threaded VideoCapture from cv2
    """

    def __init__(self, src=0):
        self.cap = cv.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.used_frame = False

    def __del__(self):
        print('Frames: {0}'.format(self.frames))

    def start(self):
        #create thread and start executing
        Thread(target=self.capture, args=()).start()
        return self

    def get_frame(self):
        return self.frame

    def capture(self):
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                self.ret, self.frame = self.cap.read()

    def stop(self):
        self.stopped = True

def threaded_read(src):
    """
    """
    capThreaded = VideoCaptureThread(src).start()
    templateImg = cv.imread('./imgs/templates/logo2.png')
    templateImg = cv.cvtColor(templateImg, cv.COLOR_BGR2GRAY)

    while True:
        frame = capThreaded.get_frame()
        if not capThreaded.ret or cv.waitKey(1) == ord("q"):
            capThreaded.stop()
            break
        imgTestedGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = search_logo_ccoeff_normed(frame, imgTestedGray, templateImg)

        win_name = 'THREADED CAPTURE'
        cv.namedWindow(win_name,cv.WINDOW_NORMAL)
        cv.resizeWindow(win_name, 500,500)
        cv.imshow(win_name, frame)
        cv.moveWindow(win_name, 0, 0)

if __name__ == '__main__':
    filename = 0
    # windowThread(filename)
    threaded_read(filename)