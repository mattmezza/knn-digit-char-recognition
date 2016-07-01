from FileSystemHelper import FileSystemHelper
from Classifier import Classifier
from Image import Image
from DigitDetector import DigitDetector
import numpy as np
import sys

RESIZED_IMAGE_WIDTH = 22
RESIZED_IMAGE_HEIGHT = 38

class Tester:
    def __init__(self, input_dir, classifier_dir, debug = False, ext = (".jpg", ".png")):
        self.testImgFolder = input_dir
        self.classifier_dir = classifier_dir
        self.fsh = FileSystemHelper(input_dir, ext)
        self.classifier = Classifier(classifier_dir, debug)
        self.classifier.train()
        self.debug = debug

    def test(self):
        self.classifier.train()
        self.fsh.for_each_file_execute_this(self.callback)

    def callback(self, file_path, file_name):
        original = Image.from_path(file_path)
        to_show = Image(original.img.copy())
        image = Image(original.img.copy()).to_gray().threshold()
        detector = DigitDetector(Image(original.img))
        contours = detector.detect()

        strFinalString = ""
        posFinalString = ""
        sepFinalString = ""
        order = 1;
        for contour in contours:
            to_show.draw_rect((contour.x, contour.y, contour.w, contour.y), (0, 255, 0), 2)

            img_roi = Image(image.img.copy()).crop(contour).resize(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT).inverte()
            npaROIResized = img_roi.vectorize(RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)
            char_discovered = self.classifier.classify(npaROIResized)
            scale = 3.0*float(contour.w)/45.0
            thickness = 3
            text_size = Image.get_text_size(char_discovered, scale, thickness)
            centered_box_x = contour.x+(contour.w-text_size[0])/2
            centered_box_y = contour.y+((contour.h-text_size[1])/2)+text_size[1]
            to_show.draw_text(char_discovered, (centered_box_x, centered_box_y), scale, (0,255,0), thickness)
            if self.debug:
                strFinalString = strFinalString + char_discovered
                posFinalString = posFinalString + str(order)
                sepFinalString = sepFinalString + "|"
            order = order+1
        if self.debug:
            print "\n" + posFinalString + "\n" + sepFinalString + "\n" + strFinalString + "\n"
        if to_show.show().wait() == 27:
            if self.debug:
                print "ESC key pressed, exiting"
            sys.exit(0)
