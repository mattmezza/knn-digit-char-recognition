import cv2

class DigitContour:

    def __init__(self, contour, boundingRect, min_contour_area = 500):
        self.contour = contour
        self.boundingRect = boundingRect
        self.min_contour_area = min_contour_area
        [self.x, self.y, self.w, self.h] = self.boundingRect
        self.area = cv2.contourArea(contour)

    def is_valid(self):
        if self.area < self.min_contour_area: return False
        aspect = 45.0 / 77.0
        charAspect = float(self.w) / float(self.h)
        error = 0.35
        minHeight = 35
        maxHeight = 256
        #We have a different aspect ratio for number 1, and it can be ~0.2
        minAspect = 0.18
        maxAspect = aspect + aspect*error;
        if charAspect > minAspect and charAspect < maxAspect and self.h >= minHeight and self.w < maxHeight:
            return True;
        else:
            return False;
