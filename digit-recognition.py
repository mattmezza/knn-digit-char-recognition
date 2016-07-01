import sys, getopt
from Teacher import Teacher
from Classifier import Classifier
from Tester import Tester

def help():
    string = 'digit-recognition.py --classifier-dir <dir> --img-dir <dir> [--teach] [--debug]'
    string = string + '\ndigit-recognition.py -c <dir> -i <dir> [-t] [-d]'
    return string

classifier_dir = ""
img_dir = ""
teach = False
debug = False
try:
    opts, args = getopt.getopt(sys.argv[1:],"c:i:td",["classifier-dir=","img-dir=","teach","debug"])
except getopt.GetoptError:
    print help()
    sys.exit(2)

if len(sys.argv)<3:
    print help()
    sys.exit(2)

for opt, arg in opts:
    if opt in ('-h', '--help'):
        print help()
        sys.exit()
    elif opt in ("-c", "--classifier-dir"):
        classifier_dir = arg
    elif opt in ("-i", "--img-dir"):
        img_dir = arg
    elif opt in ("-t", "--teach"):
        teach = True
    elif opt in ("-d", "--debug"):
        debug = True

if teach:
    teacher = Teacher(img_dir, classifier_dir, debug)
    teacher.teach()
else:
    tester = Tester(img_dir, classifier_dir, debug)
    tester.test()
