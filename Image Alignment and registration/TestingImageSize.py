from PIL import Image, ImageFilter


image1 = Image.open('../../Stefan_TrainingData_1/StefanGrid-0-.jpg')

image2 = Image.open('../../Stefan_TrainingData_1/StefanGrid-1-.jpg')

image1 = image1.resize((80, 80))
image1.show()

image2 = image2.resize((28, 28))
image2.show()



