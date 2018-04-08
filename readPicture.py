from PIL import Image
import numpy as np

im = Image.open("test_3.png")
im = im.convert('L')
arr1 = np.asarray(im).reshape(784) / 255
im = Image.open("test_8.png")
im = im.convert('L')
arr2 = np.asarray(im).reshape(784) / 255
im = Image.open("test_1.png")
im = im.convert('L')
arr3 = np.asarray(im).reshape(784) / 255
im = Image.open("test_2.png")
im = im.convert('L')
arr4 = np.asarray(im).reshape(784) / 255
#imageMatrix = np.asmatrix([arr1, arr2, arr3, arr4])
imageMatrix = np.asarray(im).reshape(1,784) / 255
print(imageMatrix)

'''test = np.matrix([[1, 2, 3], [4, 2, 1], [9, 4, 3]]);
print(np.amax(test, axis=1))'''
