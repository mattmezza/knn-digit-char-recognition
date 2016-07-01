import cv2
import numpy as np
from DigitContour import DigitContour

class Image:
    def __init__(self, original):
        self.original = original
        self.img = self.original.copy()

    def to_gray(self):
        if len(self.img.shape)>2:
            new_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.original = self.img
            self.img = new_img
        return self

    def to_color(self):
        if len(self.img.shape)<3:
            new_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            self.original = self.img
            self.img = new_img
        return self

    def blur(self, kernel_size):
        new_img = cv2.blur(self.img, kernel_size)
        self.original = self.img
        self.img = new_img
        return self

    def equalize_hist(self):
        new_img = cv2.equalizeHist(self.img)
        self.original = self.img
        self.img = new_img
        return self

    def inverte(self):
        new_img = (255-self.img)
        self.original = self.img
        self.img = new_img
        return self

    def resize(self, size):
        new_img = cv2.resize(self.img, size)
        self.original = self.img
        self.img = new_img
        return self

    def threshold(self, min_ = 0, max_ = 255, mode = cv2.THRESH_BINARY+cv2.THRESH_OTSU):
        ret2,new_img = cv2.threshold(self.img, min_, max_, mode)
        self.original = self.img
        self.img = new_img
        return self

    def draw_contours(self, contours, color = (0, 0, 255), thickness = 1):
        new_image = self.original.copy()
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            new_image = self.draw_rect(cv2.boundingRect(contour), color, thickness).img
        self.original = self.img
        self.img = new_image
        return self

    def draw_digit_contours(self, contours, color = (0, 0, 255), thickness = 1):
        new_image = self.original.copy()
        for contour in contours:
            new_image = self.draw_digit_contour(contour, color, thickness).img
        self.original = self.img
        self.img = new_image
        return self

    def draw_digit_contour(self, contour, color = (0, 0, 255), thickness = 1):
        new_image = self.draw_rect((contour.x, contour.y, contour.w, contour.h), color, thickness).img
        self.original = self.img
        self.img = new_image
        return self

    def draw_rect(self, rect, color = (0, 0, 255), thickness = 1):
        new_img = self.img.copy()
        cv2.rectangle(new_img, (rect[0], rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), color, thickness)
        self.original = self.img
        self.img = new_img
        return self

    def find_digit_contours(self):
        _, contours, hierarchy = cv2.findContours(self.img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        digit_contours = []
        for contour in contours:
            digit_contour = DigitContour(contour, cv2.boundingRect(contour))
            if digit_contour.is_valid():
                digit_contours.append(digit_contour)
        return digit_contours

    def draw_text(self, text, point, scale = .5, color = (0, 0, 255), thickness = 1):
        new_img = self.original.copy()
        cv2.putText(new_img, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        self.original = self.img
        self.img = new_img
        return self

    def fill_with(self, image, region):
        new_img = self.original.copy()
        new_img = image.img[region.y:region.y+region.h, region.x:region.x+region.w]
        self.original = self.img
        self.img = new_img
        return self

    def crop(self, rect):
        new_img = self.original.copy()
        new_img = self.original[rect.y:rect.y+rect.h, rect.x:rect.x+rect.w]
        self.original = self.img
        self.img = new_img
        return self

    def vectorize(self, length):
        return self.img.reshape((1, length))

    def resize(self, w, h):
        new_img = self.original.copy()
        new_img = cv2.resize(self.img, (w, h))
        self.original = self.img
        self.img = new_img
        return self

    def show_original(self, window_name = "original"):
        cv2.imshow(window_name, self.original)
        return self

    def show(self, window_name = "img"):
        cv2.imshow(window_name, self.img)
        return self

    def wait(self, secs = 0):
        return cv2.waitKey(secs)

    @staticmethod
    def get_text_size(text, scale = .5, thickness = 1):
        retval, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        return retval

    @staticmethod
    def destroy_img_window():
        cv2.destroyWindow("img")

    @staticmethod
    def destroy_all_windows():
        cv2.destroyAllWindows()

    @staticmethod
    def destroy_original_window():
        cv2.destroyWindow("original")

    @staticmethod
    def destroy_window(window_name):
        cv2.destroyWindow(window_name)

    @staticmethod
    def from_path(path):
        return Image(cv2.imread(path))

    @staticmethod
    def with_zeros(w, h, c = 1):
        return Image(np.zeros((w, h, c), dtype=np.uint8))
