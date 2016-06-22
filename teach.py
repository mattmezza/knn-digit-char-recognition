import sys
import numpy as np
import cv2
import os
import math
import copy

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

class Extractor:
    def __init__(self, input_dir, output_dir, withLetters = False):
        self.withLetters = withLetters
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.intClassifications = []
        self.npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    def main(self):
        self.cropped_imgs_fnames = list()
        for file_ in os.listdir(self.input_dir):
            if file_.endswith(".jpg") or file_.endswith(".png"):
                self.cropped_imgs_fnames.append(file_)
        for cropped_img_fname in self.cropped_imgs_fnames:
            self.extract_digits(cv2.imread(self.input_dir+"/"+cropped_img_fname))
            cv2.waitKey(0)

    def extract_digits(self, cropped_img):
        self.img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        #self.img_gray = cv2.equalizeHist(self.img_gray)
        self.img_gray = cv2.blur(self.img_gray, (5, 5))
        cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
        cv2.imshow("threshold", self.img_gray)
        #cv2.createTrackbar("threshold", "threshold", 207, 255, self.thresh_callback)
        self.thresh_callback(207)

    def filterContours(self, old_contours):
        new_contours = []
        avg_w = 0
        avg_h = 0
        tot_w = 0
        tot_h = 0
        for contour in old_contours:
            # create boundingRect
            x,y,w,h = cv2.boundingRect(contour)

            #cv2.rectangle(self.contours_img, (x,y), (x+w, y+h), blue, 1)
            # crop img
            sssize = w, h, 1
            cropped_digit = np.zeros(sssize, dtype=np.uint8)
            cropped_digit = self.thresholded[y:y+h, x:x+w]
            if self.verify_size(cropped_digit):
                cropped_digit = cropped_digit - 255
                self.cropped_digits.append(cropped_digit)
                new_contours.append(contour)
                tot_h = tot_h+h
                tot_w = tot_w+w
        new_new_contours = []
        numOfCont = len(new_contours)
        avg_h = 0
        avg_w = 0
        if numOfCont!=0:
            avg_h = float(tot_h)/float(numOfCont)
            avg_w = float(tot_w)/float(numOfCont)
        print "mean w and h", avg_w, avg_h
        error = .25
        for contour in new_contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w>(avg_w*(1-error)) and h>(avg_h*(1-error)) and w<(avg_w*(1+error)) and h<(avg_h*(1+error)):
                new_new_contours.append(contour)
            else:
                print "skipped contour (w,h)", w, h
        return new_new_contours

    def thresh_callback(self, threshold):
        #_ ,self.thresholded = cv2.threshold(self.img_gray, threshold, 255, cv2.THRESH_BINARY)
        ret2,self.thresholded = cv2.threshold(self.img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow("thresholded", self.thresholded)
        self.thresholded_inv = self.inverte(self.thresholded)
        _, self.contours, self.hierarchy = cv2.findContours(self.thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        self.contours_img = self.img_gray.copy()
        self.contours_img = cv2.cvtColor(self.contours_img, cv2.COLOR_GRAY2BGR)
        self.cropped_digits = []
        red = (0, 0, 255)
        blue =(255, 0, 0)
        #cv2.drawContours(self.contours_img, self.contours, -1, red, 1)
        width_max = 0
        height_max = 0

        self.filtered_contours = self.filterContours(self.contours)
        for contour in self.filtered_contours:
            x,y,w,h = cv2.boundingRect(contour)
            sssize = w, h, 1
            cropped_digit = np.zeros(sssize, dtype=np.uint8)
            cropped_digit = self.thresholded[y:y+h, x:x+w]
            self.cropped_digits.append(cropped_digit)
            cv2.rectangle(self.contours_img, (x,y), (x+w, y+h), blue, 4)

        cv2.imshow("contours", self.contours_img)
        #cv2.imshow("digits", self.all_digits)

        intChar = cv2.waitKey(0)                     # get key press

        if intChar == 13:                   # if enter key was pressed
            self.startExtraction()

    def inverte(self, imagem):
        imagem = (255-imagem)
        return imagem

    def addLetters(self, intValidChars):
        intValidChars.append(ord('a'))
        intValidChars.append(ord('s'))
        intValidChars.append(ord('d'))
        intValidChars.append(ord('f'))
        intValidChars.append(ord('g'))
        intValidChars.append(ord('h'))
        intValidChars.append(ord('j'))
        intValidChars.append(ord('k'))
        intValidChars.append(ord('l'))
        intValidChars.append(ord('y'))
        intValidChars.append(ord('x'))
        intValidChars.append(ord('c'))
        intValidChars.append(ord('v'))
        intValidChars.append(ord('b'))
        intValidChars.append(ord('n'))
        intValidChars.append(ord('m'))
        intValidChars.append(ord('q'))
        intValidChars.append(ord('w'))
        intValidChars.append(ord('e'))
        intValidChars.append(ord('r'))
        intValidChars.append(ord('t'))
        intValidChars.append(ord('z'))
        intValidChars.append(ord('u'))
        intValidChars.append(ord('i'))
        intValidChars.append(ord('o'))
        intValidChars.append(ord('p'))
        return intValidChars

    def startExtraction(self):
        intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]
        if(self.withLetters):
            intValidChars = self.addLetters(intValidChars)
        for npaContour in self.filtered_contours:                          # for each contour
            if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # if contour is big enough to consider
                [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # get and break out bounding rect
                self.contours_img = self.img_gray.copy()
                                                    # draw rectangle around each contour as we ask user for input
                cv2.rectangle(self.contours_img,           # draw rectangle on original training image
                              (intX, intY),                 # upper left corner
                              (intX+intW,intY+intH),        # lower right corner
                              (0, 0, 255),                  # red
                              4)                            # thickness

                imgROI = self.thresholded_inv[intY:intY+intH, intX:intX+intW]                                  # crop char out of threshold image
                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage

                cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference
                cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
                cv2.imshow("contours", self.contours_img)      # show training numbers image, this will now have red rectangles drawn on it

                intChar = cv2.waitKey(0)                     # get key press

                if intChar == 27:                   # if esc key was pressed
                    break                     # exit for
                elif intChar in intValidChars:      # else if the char is in the list of chars we are looking for . . .
                    self.intClassifications.append(intChar)                                                # append classification char to integer list of chars (we will convert to float later before writing to file)

                    npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                    self.npaFlattenedImages = np.append(self.npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
                # end if
            # end if
        # end for
        self.fltClassifications = np.array(self.intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats

        self.npaClassifications = self.fltClassifications.reshape((self.fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

        print "\n\ntraining complete !!\n"

        np.savetxt(self.output_dir+"/classifications.txt", self.npaClassifications)           # write flattened images to file
        np.savetxt(self.output_dir+"/flattened_images.txt", self.npaFlattenedImages)          #

        cv2.destroyAllWindows()             # remove windows from memory

    def verify_size(self, img):
        #Char sizes 45x77
    	aspect = 45.0 / 77.0
        h, w = img.shape
    	charAspect = float(w) / float(h)
    	error = 0.35
    	minHeight = 15
    	maxHeight = 256
    	#We have a different aspect ratio for number 1, and it can be ~0.2
    	minAspect = 0.18
    	maxAspect = aspect + aspect*error;
    	#area of pixels
    	area = cv2.countNonZero(img)
    	#bb area
    	bbArea = w*h;
    	#% of pixel in area
    	percPixels = area / bbArea;
        if percPixels < 0.9 and charAspect > minAspect and charAspect < maxAspect and h >= minHeight and w < maxHeight:
            return True;
        else:
        	return False;

letters = False
if sys.argv[3] and sys.argv[3]=="--with-letters":
    letters = True
extr = Extractor(sys.argv[1], sys.argv[2], letters)
extr.main()
