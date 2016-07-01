import cv2
import os
import numpy as np

class Classifier:
    def __init__(self, classifier_folder, debug = False):
        self.debug = debug
        self.classifierFolder = classifier_folder

    def train(self):
        try:
            npaClassifications = np.loadtxt(self.classifierFolder+os.path.sep+"classifications.txt", np.float32)
        except:
            print "Error, unable to open "+self.classifierFolder+os.path.sep+"classifications.txt\n"
            return
        try:
            npaFlattenedImages = np.loadtxt(self.classifierFolder+os.path.sep+"flattened_images.txt", np.float32)
        except:
            print "Error, unable to open "+self.classifierFolder+os.path.sep+"flattened_images.txt\n"
            return
        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
        self.kNearest = cv2.ml.KNearest_create()
        self.kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    def classify(self, tested):
        tested = np.float32(tested)
        retval, npaResults, neigh_resp, dists = self.kNearest.findNearest(tested, k = 1)
        res = str(chr(int(npaResults[0][0])))
        return res
