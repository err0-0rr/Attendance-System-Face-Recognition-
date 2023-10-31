# import the necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt


def erosion_util(img):
        # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # define the kernel
    kernel = np.ones((5, 5), np.uint8)
    # invert the image
    invert = cv2.bitwise_not(binr)
    # erode the image
    erosion = cv2.erode(invert, kernel,
                        iterations=1)
    plt.title("Erosion")
    # print the output
    plt.imshow(erosion, cmap='gray')
    plt.show()
    
def dialation_util(img):
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # define the kernel
    kernel = np.ones((3, 3), np.uint8)
    # invert the image
    invert = cv2.bitwise_not(binr)
    # dilate the image
    dilation = cv2.dilate(invert, kernel, iterations=1)
    plt.title("Dialation")
    # print the output
    plt.imshow(dilation, cmap='gray')
    plt.show()
  
def opening_util(img):
    # binarize the image
    binr = cv2.threshold(img, 0, 255,
                         cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      
    # define the kernel
    kernel = np.ones((3, 3), np.uint8)
      
    # opening the image
    opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN,
                               kernel, iterations=1)
    plt.title("Opening")
    # print the output
    plt.imshow(opening, cmap='gray')
    plt.show()
    
def closing_util(img):
    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      
    # define the kernel
    kernel = np.ones((3, 3), np.uint8)
      
    # opening the image
    closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1)
    plt.title("Closing")
    # print the output
    plt.imshow(closing, cmap='gray')
    plt.show()
    
    
def morphological_gradient_util(img):
    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      
    # define the kernel
    kernel = np.ones((3, 3), np.uint8)
      
    # invert the image
    invert = cv2.bitwise_not(binr)
      
    # use morph gradient
    morph_gradient = cv2.morphologyEx(invert,
                                      cv2.MORPH_GRADIENT, 
                                      kernel)
    plt.title("Morphological Gradient")
    # print the output
    plt.imshow(morph_gradient, cmap='gray')
    plt.show()


def top_hat_util(img):
    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      
    # define the kernel
    kernel = np.ones((13, 13), np.uint8)
      
    # use morph gradient
    morph_gradient = cv2.morphologyEx(binr,
                                      cv2.MORPH_TOPHAT,
                                      kernel)
    plt.title("Top Hat")
    # print the output
    plt.imshow(morph_gradient, cmap='gray')  
    plt.show()

def black_hat_util(img):
    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      
    # define the kernel
    kernel = np.ones((5, 5), np.uint8)
      
    # invert the image
    invert = cv2.bitwise_not(binr)
      
    # use morph gradient
    morph_gradient = cv2.morphologyEx(invert,
                                      cv2.MORPH_BLACKHAT,
                                      kernel)
    plt.title("Black Hat")
    # print the output
    plt.imshow(morph_gradient, cmap='gray')
    plt.show()

def sobel_edge(img):
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=27)
    plt.imshow(sobelxy,cmap = 'gray')
    plt.title('Sobel Edge Image')
    plt.show()

def canny_edge(img):
    edges = cv2.Canny(img,100,200)
    plt.imshow(edges,cmap = 'gray')
    plt.title('Cabnny edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


# read the image
img = cv2.imread("images/amrita2.tif", 0)
erosion_util(img)
dialation_util(img)
opening_util(img)
closing_util(img)
morphological_gradient_util(img)
top_hat_util(img)
black_hat_util(img)
canny_edge(img)
sobel_edge(img)
