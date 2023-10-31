import cv2
import numpy as np
import matplotlib.pyplot as plt 
from skimage.color import rgb2gray, label2rgb
from skimage import filters
from skimage.filters import gaussian
from skimage.segmentation import active_contour,chan_vese
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation import felzenszwalb


def threshold_manual(image):
    plt.figure(figsize=(15, 15))
     
    for i in range(10):
      binarized_gray = (image > (i)*0.1)*1
      plt.subplot(5,2,i+1)
      plt.title("Threshold: >"+str(round((i)*0.1,1)))
      plt.imshow(binarized_gray, cmap = 'gray')
    plt.tight_layout()

def thresholding_ski(image):
    
    plt.figure(figsize=(15, 15))
    threshold = filters.threshold_otsu(image)
    binarized_coffee = (image > threshold)*1
    plt.subplot(2,2,1)
    plt.title("Threshold: >"+str(threshold))
    plt.imshow(binarized_coffee, cmap = "gray")
    
    threshold = filters.threshold_niblack(image)
    binarized_coffee = (image > threshold)*1
    plt.subplot(2,2,2)
    plt.title("Niblack Thresholding")
    plt.imshow(binarized_coffee, cmap = "gray")
    
    threshold = filters.threshold_sauvola(image)
    plt.subplot(2,2,3)
    plt.title("Sauvola Thresholding")
    plt.imshow(threshold, cmap = "gray")
    
    
    binarized_coffee = (image > threshold)*1
    plt.subplot(2,2,4)
    plt.title("Sauvola Thresholding - Converting to 0's and 1's")
    plt.imshow(binarized_coffee, cmap = "gray")
    
    
def active_contour_util(image):  
    gray_classroom_noiseless = gaussian(image, 1)
    x1 = 400 + 100*np.cos(np.linspace(0, 2*np.pi, 300))
    x2 = 220 + 100*np.sin(np.linspace(0, 2*np.pi, 300))
    snake = np.array([x1, x2]).T
    classroom_snake = active_contour(gray_classroom_noiseless,
    								snake)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(gray_classroom_noiseless)
    plt.title("Active Contour")
    ax.plot(classroom_snake[:, 0],
    		classroom_snake[:, 1],
    		'-b', lw=5)
    ax.plot(snake[:, 0], snake[:, 1], '--r', lw=5)


def chan_vese_util(image):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    itr=100
    # Computing the Chan VESE segmentation technique
    chanvese_gray_classroom = chan_vese(image,
    									max_num_iter=itr,
    									extended_output=True)
    
    ax = axes.flatten()
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original Image")
    
    # Plotting the segmented - 100 iterations image
    ax[1].imshow(chanvese_gray_classroom[0], cmap="gray")
    title = "Chan-Vese segmentation - "+str(itr)+" iterations"
    format(len(chanvese_gray_classroom[2]))
    
    ax[1].set_title(title)
    
    # Plotting the final level set
    ax[2].imshow(chanvese_gray_classroom[1], cmap="gray")
    ax[2].set_title("Final Level Set")
    plt.show()


def mark_boundaries_util(image):
    plt.figure(figsize=(15, 15))
    classroom_segments = slic(image,
    						n_segments=50,
    						compactness=1)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(image, classroom_segments))

def simple_linear_clustering(image):
    plt.figure(figsize=(15,15))
    classroom_segments = slic(image,
    						n_segments=250,
    						compactness=10)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    
    plt.imshow(label2rgb(classroom_segments,
    					image,
    					kind = 'avg'))

def felzenszwalb_util(image):
    plt.figure(figsize=(15,15))
    classroom_segments = felzenszwalb(image,
    								scale = 2,
    								sigma=5,
    								min_size=1000)
    
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mark_boundaries(image,
    						classroom_segments))



#image = cv2.imread("images/amrita2.tif", 0)
#cv2.imshow("Original", image)

image = rgb2gray(cv2.imread("images/amrita2.tif"))
image_col=cv2.imread("images/amrita3.tif")
image_sin = rgb2gray(cv2.imread("images/amrita3.tif"))
threshold_manual(image)
thresholding_ski(image)
active_contour_util(image_sin)
chan_vese_util(image)
mark_boundaries_util(image_col)
simple_linear_clustering(image_col)
felzenszwalb_util(image_col)
