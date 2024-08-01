class Tools:
    choice = 0
    def printMenu():
        print("JZ23-2 Computer Vision")
        print("1. Image Preprocessing")
        print("2. Edge Detection")
        print("3. Shape Detection")
        print("4. Pattern Recognition")
        print("5. Exit")

    def clear():
        for i in range(32):
            print()

    def press():
        space = input("Press Enter To Continue...")