from tools import Tools
from imageprocessing import ImageProcessing as ip
from edgeDetection import EdgeDetection as ed
from shapeDetection import ShapeDetection as sd
from patternRecognition import PatternRecognition as pr


while Tools.choice != '5':
    Tools.printMenu()
    Tools.choice = input(">> ")

    if Tools.choice == '1':
        ip.threshold()
        ip.filtering()
        Tools.clear()
    elif Tools.choice == '2':
        ed.edgeDetection()
        Tools.clear()
    elif Tools.choice == '3':
        sd.shapeDetection()
        Tools.clear()
    elif Tools.choice == '4':
        pr.patternRecognition()
        Tools.clear()
    elif Tools.choice == '5':
        print("Thanks For Using")
        Tools.press()
        Tools.clear()
    else:
        print("Invalid Choice")
        Tools.press()
        Tools.clear()

