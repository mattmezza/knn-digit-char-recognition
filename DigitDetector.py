from Image import Image
import numpy as np
import cv2
import operator

class DigitDetector:
    def __init__(self, img, debug = False):
        self.original = img
        self.debug = debug
        self.thresholded = img.to_gray().blur((5, 5)).threshold()

    def detect(self):
        self.contours = self.thresholded.find_digit_contours()
        self.filtered_contours = self.filter_contours()
        self.filtered_contours.sort(key = operator.attrgetter("x"))
        return self.filtered_contours

    def filter_contours(self):
        new_contours = []
        avg_w = 0
        avg_h = 0
        tot_w = 0
        tot_h = 0
        for contour in self.contours:
            tot_h = tot_h+contour.h
            tot_w = tot_w+contour.w
        numOfCont = len(self.contours)
        avg_h = 0
        avg_w = 0
        if numOfCont!=0:
            avg_h = float(tot_h)/float(numOfCont)
            avg_w = float(tot_w)/float(numOfCont)
        if self.debug:
            print "mean w and h", avg_w, avg_h
        error = .25
        for contour in self.contours:
            x,y,w,h = contour.boundingRect
            if w>(avg_w*(1-error)) and h>(avg_h*(1-error)) and w<(avg_w*(1+error)) and h<(avg_h*(1+error)):
                new_contours.append(contour)
            else:
                if self.debug:
                    print "skipped contour (w,h)", w, h
        return new_contours
