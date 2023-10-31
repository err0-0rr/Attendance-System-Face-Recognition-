import cv2
import os
import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt 
from skimage.exposure import rescale_intensity
from scipy.signal import gaussian
import random
from skimage import filters
from skimage.segmentation import active_contour,chan_vese
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation import felzenszwalb

#Thresholding
def Thresholding(image):
    ret, sim_threshold = cv2.threshold(image,127, 255, cv2.THRESH_BINARY)
    adaptive_thre_mean =cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    adaptive_thre_mean =cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret2,otus = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.imshow("Simple Thresholding", sim_threshold)
    cv2.imshow("Adaptive Mean Thresholding ", adaptive_thre_mean)
    cv2.imshow("Adaptive Gaussian Thresholding ", adaptive_thre_mean)
    cv2.imshow("Otsu’s Thresholding ", otus)
    
#for Contrast Stretching
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2
    
def Intensity_trasformations(img):
    #Log Trasform
    c = 255/(np.log(1 + np.max(img)))
    log_transformed = c * np.log(1 + img)
    log_transformed = np.array(log_transformed, dtype = np.uint8)
    cv2.imshow("Log Trasform", log_transformed)
    
    #Power-Law (Gamma) Transformation
    for gamma in [0.1, 0.5, 1.2, 2.2]:
    # Apply gamma correction.
        gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
        cv2.imshow("Gamma Transformation with gamma: "+str(gamma), gamma_corrected)
    
    
    #Contrast Stretcing
    r1 = 70
    s1 = 0
    r2 = 140
    s2 = 255
    # Vectorize the function to apply it to each value in the Numpy array.
    pixelVal_vec = np.vectorize(pixelVal)
    # Apply contrast stretching.
    contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)
    cv2.imshow("Contrast_stretch", contrast_stretched)
    
    
def Intensity_level_slicing(img, min_range, max_range):
   
   # Find width and height of image
   row, column = img.shape
   # Create an zeros array to store the sliced image
   img1 = np.zeros((row,column),dtype = 'uint8')


    # Loop over the input image and if pixel value lies in desired range set it to 255 otherwise set it to 0.
   for i in range(row):
       for j in range(column):
           if img[i,j]>min_range and img[i,j]<max_range:
               img1[i,j] = 255
           else:
               img1[i,j] = 0
            # Display the image
   cv2.imshow('sliced image', img1)
   
#Bit planes
def Bit_Plane_Slicing(img):
    
    #Iterate over each pixel and change pixel value to binary using np.binary_repr() and store it in a list.
    lst = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lst.append(np.binary_repr(img[i][j] ,width=8)) # width = no. of bits

    # We have a list of strings where each string represents binary pixel value. To extract bit planes we need to iterate over the strings and store the characters corresponding to bit planes into lists.
    # Multiply with 2^(n-1) and reshape to reconstruct the bit image.
    eight_bit_img = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(img.shape[0],img.shape[1])
    seven_bit_img = (np.array([int(i[1]) for i in lst],dtype = np.uint8) * 64).reshape(img.shape[0],img.shape[1])
    six_bit_img = (np.array([int(i[2]) for i in lst],dtype = np.uint8) * 32).reshape(img.shape[0],img.shape[1])
    five_bit_img = (np.array([int(i[3]) for i in lst],dtype = np.uint8) * 16).reshape(img.shape[0],img.shape[1])
    four_bit_img = (np.array([int(i[4]) for i in lst],dtype = np.uint8) * 8).reshape(img.shape[0],img.shape[1])
    three_bit_img = (np.array([int(i[5]) for i in lst],dtype = np.uint8) * 4).reshape(img.shape[0],img.shape[1])
    two_bit_img = (np.array([int(i[6]) for i in lst],dtype = np.uint8) * 2).reshape(img.shape[0],img.shape[1])
    one_bit_img = (np.array([int(i[7]) for i in lst],dtype = np.uint8) * 1).reshape(img.shape[0],img.shape[1])

    #Concatenate these images for ease of display using cv2.hconcat()
    finalr = cv2.hconcat([eight_bit_img,seven_bit_img,six_bit_img,five_bit_img])
    finalv =cv2.hconcat([four_bit_img,three_bit_img,two_bit_img,one_bit_img])
    # Vertically concatenate
    final = cv2.vconcat([finalr,finalv])
    # Display the images
    cv2.imshow('Bit Planes',final)
    #plt.imshow(eight_bit_img) 
    #plt.show()


#Histogram Equalization
def Histogram_equ(img):
    
    #convert to NumPy array
    img_array = np.asarray(img)

    #STEP 1: Normalized cumulative histogram
    #flatten image array and calculate histogram via binning
    histogram_array = np.bincount(img_array.flatten(), minlength=256)
    #normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array/num_pixels
    #normalized cumulative histogram
    chistogram_array = np.cumsum(histogram_array)
    #STEP 2: Pixel mapping lookup table
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
    #STEP 3: Transformation
    # flatten image array into 1D list
    img_list = list(img_array.flatten())
    # transform pixel values to equalize
    eq_img_list = [transform_map[p] for p in img_list]
    # reshape and write back into img_array
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
    cv2.imshow('Histogam Equivalization', eq_img_array)
  
#For Histogram Specification(matching)

def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, -1)
    # We need to mask the negative differences
    # since we are looking for values above
    if np.all(mask):
        c = np.abs(diff).argmin()
        return c # returns min index of the nearest if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

def hist_match(original, specified):

    oldshape = original.shape
    original = original.ravel()
    specified = specified.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(original, return_inverse=True,return_counts=True)
    t_values, t_counts = np.unique(specified, return_counts=True)

    # Calculate s_k for original image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    # Calculate s_k for specified image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Round the values
    sour = np.around(s_quantiles*255)
    temp = np.around(t_quantiles*255)
    
    # Map the rounded values
    b=[]
    for data in sour[:]:
        b.append(find_nearest_above(temp,data))
    b= np.array(b,dtype='uint8')

    return b[bin_idx].reshape(oldshape)

def Histogram_match(img, img_ref):
    original = img
    specified = img_ref

    # perform Histogram Matching
    a = hist_match(original, specified)

    # Display the images
    plt.imshow(original) 
    plt.show()
    #display the histogram
    hist,bins = np.histogram(original.flatten(),256,[0,256])
    plt.hist(original.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.show()

    plt.imshow(specified) 
    plt.show()
    #display the histogram
    hist,bins = np.histogram(specified.flatten(),256,[0,256])
    plt.hist(specified.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.show()

    plt.imshow(a) 
    plt.show()
    # display the histogram
    hist,bins = np.histogram(a.flatten(),256,[0,256])
    plt.hist(a.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.show()

#convolutin
def convolve(image, kernel):

	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
    
    # loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()
			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k
            
            # rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	# return the output image
	return output

def Smoothing(img):
   
    # construct average blurring kernels used to smooth an image
    smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
    largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
    # construct a sharpening filter
    sharpen = np.array(([0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]), dtype="int")
    # construct the Laplacian kernel used to detect edge-like regions of an image
    laplacian = np.array(([0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]), dtype="int")
    # construct the Sobel x-axis kernel
    sobelX = np.array(([-1, 0, 1],
                      [-2, 0, 2],
                          [-1, 0, 1]), dtype="int")
    # construct the Sobel y-axis kernel
    sobelY = np.array(([-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]), dtype="int")
    # construct the kernel bank, a list of kernels we're going
    # to apply using both our custom `convole` function and
    # OpenCV's `filter2D` function
    kernelBank = (
        ("small_blur", smallBlur),
        ("large_blur", largeBlur),
        ("sharpen", sharpen),
        ("laplacian", laplacian),
        ("sobel_x", sobelX),
        ("sobel_y", sobelY)
    )
    # load the input image and convert it to grayscale
    image = img
    gray = image
    # loop over the kernels
    for (kernelName, kernel) in kernelBank:
    	# apply the kernel to the grayscale image using both
        # our custom `convole` function and OpenCV's `filter2D`
    	# function
    	print("[INFO] applying {} kernel".format(kernelName))
    	convoleOutput = convolve(gray, kernel)
    	#opencvOutput = cv2.filter2D(gray, -1, kernel)
    	# show the output images
    	cv2.imshow("original", gray)
    	cv2.imshow("{} - convole".format(kernelName), convoleOutput)
    	#cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
        

# median filter
def Median_filter(data):
    filter_size=3
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
            
    data_final = (data_final).astype("uint8")
    cv2.imshow("Median Filter", data_final)

def Mean_filter(image):
    img = image
    w = 2

    for i in range(2,image.shape[0]-2):
        for j in range(2,image.shape[1]-2):
            block = image[i-w:i+w+1, j-w:j+w+1]
            m = np.mean(block,dtype=np.float32)
            img[i][j] = int(m)
    img = (img).astype("uint8")
    cv2.imshow("Mean Filter", img)


def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

def wiener_filter(img, K=10):
    
    kernel = gaussian_kernel(3)
    
    dummy = np.copy(img)
    kernel = np.pad(kernel, [(0, dummy.shape[0] - kernel.shape[0]), (0, dummy.shape[1] - kernel.shape[1])], 'constant')
    # Fourier Transform
    dummy = fft2(dummy)
    kernel = fft2(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    dummy = (dummy).astype("uint8")
    cv2.imshow("Wiener Filter", dummy)

def gaussian_img(x,sigma):
    return (1.0/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return np.sqrt(np.abs((x1-x2)**2-(y1-y2)**2))

def bilateral_filter(image, diameter, sigma_i, sigma_s):
    new_image = np.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x =row - (diameter/2 - k)
                    n_y =col - (diameter/2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    gi = gaussian_img((image[int(n_x)][int(n_y)] - image[row][col]), sigma_i)
                    gs = gaussian_img(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi * gs
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(np.round(filtered_image))
    new_image = (new_image).astype("uint8")
    cv2.imshow("Bilateral Filter", new_image)
def Bilateral_filter(image):
    #bilateral_filter(image, 7, 20.0, 20.0)
    new_image=cv2.bilateralFilter(image, 7, 20.0, 20.0)
    cv2.imshow("Bilateral Filter", new_image)
    
    
def Geometric_mean(img):
    rows, cols = img.shape[:2]
    ksize = 5
    padsize = int((ksize-1)/2)
    pad_img = cv2.copyMakeBorder(img, *[padsize]*4, cv2.BORDER_DEFAULT)
    geomean1 = np.zeros_like(img)
    for r in range(rows):
        for c in range(cols):
            geomean1[r, c] = np.prod(pad_img[r:r+ksize, c:c+ksize])**(1/(ksize**2))
    geomean1 = np.uint8(geomean1)
    cv2.imshow('Geometric_mean filter', geomean1)

def get_kernel():
    return np.ones((3, 3), np.float32) / 9

def get_mean_with_kernel(filter_area, kernel):
    # Fastest solution to multiply the matrices and get the result.
    return np.sum(np.multiply(kernel, filter_area))
    """
    This is also slower, since it requires this operation to be done for each of the channels.
    average = 0
    for i in range(3):
        average += np.sum(np.multiply(kernel, filter_area))
        
    return average
    """
    """
    Eliminated this loop based averaging, since it takes too much time.
    for krow in range(kernel_height):
        for kcol in range(kernel_width):
            row_index = row - (krow - middle_point)
            col_index = column - (kcol - middle_point)
            average_value += image[row_index][col_index][channel] * kernel[krow][kcol]
    return average_value
    """

def get_median(filter_area):
    res = np.median(filter_area)
    return res

def mean_median_balanced_filter(img):
    image = img.copy()
    height, width = image.shape[:2]
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    BALANCE_ALPHA = 0.2
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            filter_area = image[row - 1:row + 2, column - 1:column + 2]
            mean_filter_vector = get_mean_with_kernel(filter_area, get_kernel())
            median_filter_vector = get_median(filter_area)
            image[row][column] = BALANCE_ALPHA * mean_filter_vector + (1 - BALANCE_ALPHA) * median_filter_vector
    cv2.imshow("Mean_median_balanced_filter", image)

def minimumBoxFilter(img):
  n=3

  # Creates the shape of the kernel
  size = (n, n)
  shape = cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(shape, size)

  # Applies the minimum filter with kernel NxN
  imgResult = cv2.erode(img, kernel)

  # Shows the result
  cv2.imshow('Minbox filter ', imgResult)


def maximumBoxFilter(img): 
  # Creates the shape of the kernel
  n=3
  size = (n,n)
  shape = cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(shape, size)

  # Applies the maximum filter with kernel NxN
  imgResult = cv2.dilate(img, kernel)

  # Shows the result
  cv2.imshow('Maxbox filter ', imgResult)



def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0

    return H

def notch_reject(img):
    img_shape = img.shape

    original = np.fft.fft2(img) 
    center = np.fft.fftshift(original)  
    
    NotchRejectCenter = center * notch_reject_filter(img_shape, 32, 50, 50)  
    NotchReject = np.fft.ifftshift(NotchRejectCenter)
    inverse_NotchReject = np.fft.ifft2(NotchReject)  # Compute the inverse DFT of the result
    plot_image =  np.abs(inverse_NotchReject)
    plot_image = (plot_image).astype("uint8")
    cv2.imshow("Notch reject filter", plot_image)
    
    
    
def fastDenoise(image):
    noiseless_image_bw = cv2.fastNlMeansDenoising(image, None, 20, 7, 21) 
    img=np.asarray(noiseless_image_bw)
    cv2.imshow("fast", img)


def add_salt_pepper_noise(image):
 
    # Getting the dimensions of the image
    img = image.copy()
    row , col = img.shape
     
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to white
        img[y_coord][x_coord] = 255
         
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        img[y_coord][x_coord] = 0
         
    return img
frame=0
def add_gauss_noise(img):
    gauss_noise=np.zeros(img.shape,dtype=np.uint8)
    cv2.randn(gauss_noise,128,20)
    gauss_noise=(gauss_noise*0.5).astype(np.uint8)
    gn_img=cv2.add(img,gauss_noise)
    return gn_img

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
    plt.imshow(classroom_segments)

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

def sobel_edge(img):
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=27)
    plt.imshow(sobelxy,cmap = 'gray')
    plt.title('Sobel Edge Image')
    plt.show()

def canny_edge(img):
    edges = cv2.Canny(img,100,200)
    plt.subplot(1, 2, 1)
    plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
def enhance_preset(img, situation):
    #Thresholding(image)
    #Intensity_trasformations(image)
    #Intensity_level_slicing(image, 10, 60)
    #Bit_Plane_Slicing(image)
    #Histogram_equ(image)
    #Histogram_match(image, image_ref)
    #Smoothing(image)
        #blur
        #sharpen
        #laplasian
        #sobelx
        #sobely
        
    #Median_filter(image)
    #Mean_filter(image)
    #wiener_filter(image, 3)
    #Bilateral_filter(image)
    #minimumBoxFilter(image)
    #maximumBoxFilter(image)
    #mean_median_balanced_filter(image)
    #notch_reject(image)
    #fastDenoise(image)
    
    #threshold_manual(image)
    #thresholding_ski(image)
    #active_contour_util(image_sin)
    #chan_vese_util(image)
    #mark_boundaries_util(image_col)
    #simple_linear_clustering(image_col)
    #felzenszwalb_util(image_col)
    
    #erosion_util(img)
    #dialation_util(img)
    #opening_util(img)
    #closing_util(img)
    #morphological_gradient_util(img)
    #top_hat_util(img)
    #black_hat_util(img)
    #canny_edge(img)
    #sobel_edge(img)
    if situation=='indoor.':
        img=Bit_Plane_Slicing(img)
        img=Histogram_equ(img)
        img=mean_median_balanced_filter(img)
        img=canny_edge(img)
        img=opening_util(img)
        img=thresholding_ski(img)
        
    elif situation=='night':
        img=Median_filter(img)
        img=Bilateral_filter(img)
        img=fastDenoise()
        img=sobel_edge(img)
        img=simple_linear_clustering(img)
        img=Intensity_level_slicing(img, 70, 200)
        
    return img
        
    

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)



video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    frame=enhance_preset(frame, 'indoor')
    
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frames)
    print(len(faces))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()