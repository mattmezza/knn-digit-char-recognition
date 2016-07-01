from FileSystemHelper import FileSystemHelper
from DigitDetector import DigitDetector
from Image import Image
import numpy as np

RESIZED_IMAGE_WIDTH = 22
RESIZED_IMAGE_HEIGHT = 38
valid_keys = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]

class Teacher:
    def __init__(self, input_dir, output_dir, debug = False, ext = (".jpg", ".png")):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.debug = debug
        self.intClassifications = []
        self.npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        self.fsh = FileSystemHelper(input_dir, ext)

    def teach(self):
        self.fsh.for_each_file_execute_this(self.callback)
        self.fltClassifications = np.array(self.intClassifications, np.float32)
        self.npaClassifications = self.fltClassifications.reshape((self.fltClassifications.size, 1))
        if self.debug:
            print "Training complete!"
        np.savetxt(self.output_dir+"/classifications.txt", self.npaClassifications)
        np.savetxt(self.output_dir+"/flattened_images.txt", self.npaFlattenedImages)

    def callback(self, file_path, file_name):
        original = Image.from_path(file_path)
        image = Image(original.img.copy())
        detector = DigitDetector(image)
        contours = detector.detect()
        key_typed = image.to_color().draw_digit_contours(contours).show().wait()
        if key_typed==27:
            if self.debug:
                print "exiting..."
            exit()
        elif key_typed==13:
            for contour in contours:
                cropped = Image.with_zeros(contour.w, contour.h).fill_with(original.to_gray(), contour)
                digit_typed = cropped.show("cropped").wait()

                if digit_typed in valid_keys:
                    self.intClassifications.append(digit_typed)
                    flattened_image = cropped.resize(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT).img.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                    self.npaFlattenedImages = np.append(self.npaFlattenedImages, flattened_image, 0)
                elif digit_typed==27:
                    if self.debug:
                        print "exiting..."
                    exit()
                else:
                    if self.debug:
                        print "skipping this digit"
        Image.destroy_window("cropped")
