# TrainAndTest.py

import cv2
import numpy as np
import operator
import os
import sys

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        aspect = 45.0 / 77.0
        charAspect = float(self.intRectWidth) / float(self.intRectHeight)
        error = 0.35
        minHeight = 35
        maxHeight = 256
        #We have a different aspect ratio for number 1, and it can be ~0.2
        minAspect = 0.18
        maxAspect = aspect + aspect*error;
        if charAspect > minAspect and charAspect < maxAspect and self.intRectHeight >= minHeight and self.intRectWidth < maxHeight:
            return True;
        else:
            return False;

###################################################################################################

class Tester:
    def __init__(self, testImgFolder, classifierFolder):
        self.testImgFolder = testImgFolder
        self.classifierFolder = classifierFolder
        self.allContoursWithData = []                # declare empty lists,
        self.validContoursWithData = []              # we will fill these shortly
        try:
            npaClassifications = np.loadtxt(self.classifierFolder+"/classifications.txt", np.float32)                  # read in training classifications
        except:
            print "error, unable to open classifications.txt, exiting program\n"
            os.system("pause")
            return
        try:
            npaFlattenedImages = np.loadtxt(self.classifierFolder+"/flattened_images.txt", np.float32)                 # read in training images
        except:
            print "error, unable to open flattened_images.txt, exiting program\n"
            os.system("pause")
            return
        self.npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train
        self.kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object
        self.kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        self.cropped_imgs_fnames = list()
    def test(self):
        for file_ in os.listdir(self.testImgFolder):
            if file_.endswith(".jpg") or file_.endswith(".png"):
                self.cropped_imgs_fnames.append(file_)
        for cropped_img_fname in self.cropped_imgs_fnames:
            self.testImg(cv2.imread(self.testImgFolder+"/"+cropped_img_fname))
            if cv2.waitKey(0) == 27:
                break
        cv2.destroyAllWindows()             # remove windows from memory

    def testImg(self, imgTestingNumbers):
        self.imgTestingNumbers = imgTestingNumbers
        self.original = self.imgTestingNumbers.copy()
        if self.imgTestingNumbers is None:                           # if image was not read successfully
            print "error: image not read from file \n\n"        # print error message to std out
            os.system("pause")                                  # pause so user can see error message
            return                                              # and exit function (which exits program)
        # end if

        self.imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
        #self.imgEq = cv2.equalizeHist(self.imgGray)
        self.imgBlurred = cv2.GaussianBlur(self.imgGray, (5,5), 0)                    # blur
        cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
        cv2.imshow("threshold", self.imgGray)
        #cv2.createTrackbar("threshold", "threshold", 207, 255, self.thresh_callback)
        self.thresh_callback(207)

    def thresh_callback(self, threshold):
        self.allContoursWithData = []
        self.validContoursWithData = []

        #_ ,self.thresholded = cv2.threshold(self.imgBlurred, threshold, 255, cv2.THRESH_BINARY)
        ret2,self.thresholded = cv2.threshold(self.imgBlurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.imgTestingNumbers = self.thresholded.copy()
        self.imgTestingNumbers = cv2.cvtColor(self.imgTestingNumbers, cv2.COLOR_GRAY2BGR)
        self.imgThresh = self.thresholded.copy()
        imgThreshCopy = self.thresholded.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
        imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                     cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                     cv2.CHAIN_APPROX_NONE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

        for npaContour in npaContours:                             # for each contour
            contourWithData = ContourWithData()                                             # instantiate a contour with data object
            contourWithData.npaContour = npaContour                                         # assign contour to contour with data
            contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
            contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
            contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
            self.allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
        # end for
        for contourWithData in self.allContoursWithData:                 # for all contours
            if contourWithData.checkIfContourIsValid():             # check if valid
                self.validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        self.validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right
        strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program
        posFinalString = ""
        sepFinalString = ""
        order = 1;
        new_contours = []
        tot_w = 0
        tot_h = 0
        for contourWithData in self.validContoursWithData:
            w = contourWithData.intRectWidth
            h = contourWithData.intRectHeight
            tot_w = tot_w + w
            tot_h = tot_h + h
        numOfCont = len(self.validContoursWithData)
        avg_h = 0
        avg_w = 0
        if numOfCont!=0:
            avg_h = float(tot_h)/float(numOfCont)
            avg_w = float(tot_w)/float(numOfCont)
        error = .25
        for contourWithData in self.validContoursWithData:
            w = contourWithData.intRectWidth
            h = contourWithData.intRectHeight
            if w>(avg_w*(1-error)) and h>(avg_h*(1-error)) and w<(avg_w*(1+error)) and h<(avg_h*(1+error)):
                new_contours.append(contourWithData)
            else:
                print "skipped contour (w,h)", w, h

        for contourWithData in new_contours:
            cv2.rectangle(self.imgTestingNumbers,                                        # draw rectangle on original testing image
                          (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                          (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                          (0, 255, 0),              # green
                          2)                        # thickness
            cv2.putText(self.imgTestingNumbers, str(order), (contourWithData.intRectX, contourWithData.intRectY), cv2.FONT_HERSHEY_SIMPLEX, .5, 255, 2)
            imgROI = self.imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                               contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
            imgROI = self.inverte(imgROI)
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array
            npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats
            retval, npaResults, neigh_resp, dists = self.kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest
            strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results
            strFinalString = strFinalString + strCurrentChar            # append current char to full string
            posFinalString = posFinalString + str(order)
            sepFinalString = sepFinalString + "|"
            order = order+1
        # end for
        print "\n" + posFinalString + "\n" + sepFinalString + "\n" + strFinalString + "\n"                 # show the full string
        cv2.imshow("imgTestingNumbers", self.imgTestingNumbers)      # show input image with green boxes drawn around found digits

    def inverte(self, imagem):
        imagem = (255-imagem)
        return imagem

###################################################################################################
if __name__ == "__main__":
    #main()
    tester = Tester(sys.argv[1], sys.argv[2])
    tester.test()
# end if
