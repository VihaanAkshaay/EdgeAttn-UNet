'''
This file has all the edge algorithms in it (borrowed from HW2)
'''
import numpy as np
import math 
from scipy.signal import convolve2d

def gaussian_kernel(sigma, truncate=3):
    """Gaussian kernel

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian
    truncate : float
        Truncate the kernel at `truncate` standard deviations.
        Default: 3 standard deviations

    Returns
    -------
    gaussian_kernel : np.ndarray
        A (truncate*sigma, truncat*sigma) 2D NumPy array with the Gaussian
        kernel. Number of pixels are rounded up to the nearest odd integer.
    """

    #kernel size
    k = sigma*truncate

    #Gaussial_filter
    gaussian_filter = np.zeros((k,k), np.float32)
    center_x = (k//2)
    center_y = (k//2)

    coeff = 1/(2*math.pi*math.pow(sigma,2))
    
    for i in range(k):
      for j in range(k):
        gaussian_filter[i,j] = coeff * math.exp(-(math.pow(center_x - i,2)+math.pow(center_y-j,2))/(2*math.pow(sigma,2)))

    gaussian_filter = gaussian_filter/np.sum(gaussian_filter)
    return gaussian_filter


# generates the laplacian of a gaussian 
def LoG_kernel(sigma):
  """ Laplacian of a Gaussian calculated by 
  first, defining the gaussian with the sigma from input,
  convolved with a laplacian operator to generate LoG

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian

    Returns
    -------
    LoG : np.ndarray
        A 2D NumPy array with the Laplacian of Gaussian
        kernel using the definition above.
  """
  laplacian_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
  LoG = convolve2d(gaussian_kernel(sigma), laplacian_kernel, boundary='wrap', mode='same')

  return LoG


def zero_crossings(img, threshold=0):
    """Find the zero crossings in an image

    Parameters
    ----------
    img : np.ndarray
        A 2D NumPy array for which zero crossings should be found
    threshold : float, optional
        Minimum difference of neighboring gray values required to consider
        the zero crossing an edge

    Returns
    -------
    zero_cross : np.ndarray
        An image of the same size as `img` with zero crossings=1, all other
        pixels=0
    """
    zero_cross = np.zeros_like(img)
    
    for i in range(1,img.shape[0]-1):
      for j in range(1,img.shape[1]-1):

        #Left and Right?
        if (img[i-1,j]*img[i+1,j]<=0) and (abs(img[i-1,j] - img[i+1,j]) >= threshold):
          zero_cross[i,j] = 1
        
        #Up and Down?
        if (img[i,j-1]*img[i,j+1]<=0) and (abs(img[i,j-1] - img[i,j+1]) >= threshold):
          zero_cross[i,j] = 1

        #Right-up and Left-down
        if (img[i-1,j-1]*img[i+1,j+1]<=0) and (abs(img[i-1,j-1] - img[i+1,j+1]) >= threshold):
          zero_cross[i,j] = 1

        #Right-down and Left-up
        if (img[i+1,j-1]*img[i-1,j+1]<=0) and (abs(img[i+1,j-1] - img[i-1,j+1]) >= threshold):
          zero_cross[i,j] = 1


    return zero_cross


def inv_zero_crossings(img, threshold=0):
    """Find the zero crossings in an image

    Parameters
    ----------
    img : np.ndarray
        A 2D NumPy array for which zero crossings should be found
    threshold : float, optional
        Minimum difference of neighboring gray values required to consider
        the zero crossing an edge

    Returns
    -------
    zero_cross : np.ndarray
        An image of the same size as `img` with zero crossings=1, all other
        pixels=0
    """
    zero_cross = np.ones_like(img)
    
    for i in range(1,img.shape[0]-1):
      for j in range(1,img.shape[1]-1):

        #Left and Right?
        if (img[i-1,j]*img[i+1,j]<=0) and (abs(img[i-1,j] - img[i+1,j]) >= threshold):
          zero_cross[i,j] = 0
        
        #Up and Down?
        if (img[i,j-1]*img[i,j+1]<=0) and (abs(img[i,j-1] - img[i,j+1]) >= threshold):
          zero_cross[i,j] = 0

        #Right-up and Left-down
        if (img[i-1,j-1]*img[i+1,j+1]<=0) and (abs(img[i-1,j-1] - img[i+1,j+1]) >= threshold):
          zero_cross[i,j] = 0

        #Right-down and Left-up
        if (img[i+1,j-1]*img[i-1,j+1]<=0) and (abs(img[i+1,j-1] - img[i-1,j+1]) >= threshold):
          zero_cross[i,j] = 0


    return zero_cross

def MH_edge(img,threshold,sigma):
    '''
    Uses all the above methods to finally generate the image that returns the edge.

    Parameters
    ----------
    img : np.ndarray
        A 2D NumPy array for which zero crossings should be found
    threshold : float, optional (For zero_crossing module)
        Minimum difference of neighboring gray values required to consider
        the zero crossing an edge
    sigma : float (For LoG module)
        Standard deviation of the Gaussian

    Returns
    -------

    zero_cross : np.ndarray
        An image of the same size as `img` with zero crossings=1, all other pixels=0
    
    '''

    altered_image = convolve2d(img, LoG_kernel(sigma), boundary='wrap', mode='same')
    return zero_crossings(altered_image,threshold)

def MH_edge_inv(img,threshold,sigma):
    '''
    Uses all the above methods to finally generate the image that returns the edge.

    Parameters
    ----------
    img : np.ndarray
        A 2D NumPy array for which zero crossings should be found
    threshold : float, optional (For zero_crossing module)
        Minimum difference of neighboring gray values required to consider
        the zero crossing an edge
    sigma : float (For LoG module)
        Standard deviation of the Gaussian

    Returns
    -------

    zero_cross : np.ndarray
        An image of the same size as `img` with zero crossings=1, all other pixels=0
    
    '''

    altered_image = convolve2d(img, LoG_kernel(sigma), boundary='wrap', mode='same')
    return inv_zero_crossings(altered_image,threshold)