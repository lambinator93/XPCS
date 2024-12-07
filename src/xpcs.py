# xpcs.py

### Libraries ###

import numpy as np
import h5py
import hdf5plugin
import time
import os
from scipy.optimize import curve_fit
from scipy.special import erfinv
from scipy import constants as sc
import scipy.io
import ipywidgets as widgets
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

from PIL import Image, ImageSequence


### For Data Analysis ###

#g2_exp = g2(q,dt)-1 = A*np.exp(-(t/tt)**b)

def gaussian(x, A, mu, sigma, c):
    return A*np.exp(-0.5 * ((x - mu)/sigma)**2) + c

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, background):
    x, y = xy
    #x0 = float(x0)
    #y0 = float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = background + amplitude*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    return g.ravel()

def g2_exp(t,A,C,tt,b):
    return C+A*np.exp(-(t/tt)**b)

def norm_g2(t,tt,b):
    return np.exp(-(t/tt)**b)

def g2_two_tau(t,C,A,a,tt1,tt2,b):
    return C+A*np.abs(a*np.exp(-(t/tt1)**b)+(1-a)*np.exp(-(t/tt2)**b))**2

def g2_exp_two(t,C,A1,A2,tt1,tt2,b1,b2):
    return C+A1*np.exp(-(t/tt1)**b1)+A2*np.exp(-(t/tt2)**b2)

def tau(q,A,c,a):
    return A*np.exp(-(q/c))

def raleigh(lam,fl,dl):
    return 1.22*fl*lam/dl

def newtth(Ein,Ebl,Th1):
    
    #Commonly used XRD energies in keV
    Cu = 8.0478
    Fe = 6.3998
    Co = 6.9257
    Mo = 17.45
    
    if isinstance(Ein, str):
        if Ein == 'Mo':
            Ein = None 
            Ein = 17.45
        if Ein == 'Cu':
            Ein = None 
            Ein = 8.0478
        if Ein == 'Co':
            Ein = None 
            Ein = 6.9257
        if Ein == 'Fe':
            Ein = None 
            Ein = 6.3998
 
    #Energies are in keV, 2Thetas in degrees
    ThNew = np.arcsin((Ein/Ebl)*np.sin((Th1/2)*np.pi/180))*180/np.pi
    #print(Th2*2)
    return 2*ThNew

def th2q(En,tth):
    wl = 1.2398*10/En
    return (4*np.pi/wl)*np.sin(tth*np.pi/(180*2))

def beta_Michelson(arr):
    I_min = np.min(arr)
    I_max = np.max(arr)
    return (I_max-I_min)/(I_max+I_min)

def beta(arr):
    return np.std(arr)/(np.mean(arr))

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def binning(det,binSize,pixSize=0.075,mode='mean'):
    # To bin the 800x800 to 400x400, we first reshape the array
    # Now, we take the mean across the newly introduced dimensions (2 and 4, the binning dimensions)
    # This averages every 2x2 bin into a single value, effectively reducing the resolution
    # Also returns the binned pixel sizes 
    if len(det.shape)==2:
        
        if mode == 'mean':
            binnedDet = (det.reshape(int(det.shape[0]//binSize), binSize, int(det.shape[1]//binSize),binSize)).mean(axis=(1, 3))

        if mode == 'sum':
            binnedDet = (det.reshape(int(det.shape[0]//binSize), binSize, int(det.shape[1]//binSize), binSize)).sum(axis=(1, 3))
            
    if len(det.shape)==3:
        
        if mode == 'mean':
            binnedDet = (det.reshape(det.shape[0], int(det.shape[1]//binSize), binSize, int(det.shape[2]/binSize),binSize)).mean(axis=(2, 4))

        if mode == 'sum':
            binnedDet = (det.reshape(det.shape[0], int(det.shape[1]//binSize), binSize, int(det.shape[2]//binSize), binSize)).sum(axis=(2, 4))
    
    return binnedDet, pixSize*binSize

### Creates Elliptical Masks ###

def const_int_mask(arr,sigmax,sigmay,sigAll = 3,num_rings=5,num_slices=10,tol=0.20,res=0.1):
    """
    Creates ROI masks of ellipses of equal probability rings. Past attempts have used erfinv but this one will integrate the    intensity up to the 3rd std and calculate the percentage for each ring interatively. First it will find the appropriate ring "widths" by this procedure. 
    
    Inputs:
    arr: Cropped detector array (nframes, height, width)
    sigmax: Standard deviation in the x-direction in pixels from Gaussian fit
    sigmay: Standard deviation in the y-direction in pixels from Gaussian fit
    
    
    """
    # Parameters
    _, height, width = arr.shape  # Image dimensions
    
    center = (height // 2, width // 2)  # Center of the image
    
    '''''
    #Addresses strange issue of 1 being the last number or not
    #Outer boundary for rings of equal probabilities in fractions of standard deviations
    probs = np.arange(1/num_rings,1,1/num_rings)
    if probs[-1]>0.99 and probs[-1]<=1:
        probs[-1] = 0.998  
    else:
        probs = np.append(np.arange(1/num_rings,1,1/num_rings),0.998)
        
    
    #ring_widths = 1/np.sqrt(probs)
    #ring_widths = np.sqrt(2)*erfinv(probs)
    #print(ring_widths)
    '''''
        
    # Create a grid of coordinates
    y, x = np.ogrid[:height, :width]

    ### First we must calculate the ring widths of constant intensity ###
    # Calculate the elliptical distance from the center
    elliptical_distance = np.sqrt((x - center[1])**2/(sigmax)**2 + (y - center[0])**2/(sigmay)**2)
    
    total_peak_mask = (elliptical_distance >= 0) & (elliptical_distance < sigAll) #Mask for integrating the whole peak intensity, 3std
    
    avg_img = np.sum(arr[0:-1,:,:],axis=0) #Take the mean of the first 10 images 
    
    total_int = np.sum(avg_img[total_peak_mask==1]) #Get the total intensity of the whole peak
    
    ring_widths = np.zeros(num_rings)
    
    for i in range(num_rings):
        
        integ_int = 0
        
        while not((integ_int/total_int) <= (1/num_rings)*(1+tol) and (integ_int/total_int) >= (1/num_rings)*(1-tol)):
            
            ring_widths[i] += res

            # Choose ellipse rings
            if i==0:
                inner_radius = 0
                outer_radius = ring_widths[i]
            else:
                inner_radius = ring_widths[i-1]
                outer_radius = ring_widths[i]
             
            
            temp_mask = (elliptical_distance >= inner_radius) & (elliptical_distance < outer_radius) #Create maske
            
            integ_int = np.sum(avg_img[temp_mask==1]) #Get the total intensity of the whole peak
    
    print(ring_widths)
    

    # Calculate the angle of each pixel relative to the center
    angle_from_center = np.arctan2(y - center[0], x - center[1])  # Angle in radians
    angle_from_center = (angle_from_center + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2*pi)

    # Define the boundaries for the elliptical rings
    #ring_width = 1.0 / num_rings  # Each ring will occupy a fraction of the elliptical distance range
    roi_mask = np.zeros((num_rings*num_slices,height, width))  # Create a blank mask for all ROIs
    #roi_mask[:,:,:] = -1 #This is for processing later. -1 is an unphysical value, unlike 0
    
    # Assign each ROI a unique index for visualization
    roi_index = 0
    
    for i in range(num_rings):
        # Choose ellipse rings
        if i==0:
            inner_radius = 0
            outer_radius = ring_widths[i]
        else:
            inner_radius = ring_widths[i-1]
            outer_radius = ring_widths[i]
        
        # Create a mask for each elliptical ring
        ring_mask = (elliptical_distance >= inner_radius) & (elliptical_distance < outer_radius)

        for j in range(num_slices):
            # Define angular boundaries for each segment
            start_angle = j * (2 * np.pi / num_slices)
            end_angle = (j + 1) * (2 * np.pi / num_slices)
            # Create a mask for each angular segment
            slice_mask = (angle_from_center >= start_angle) & (angle_from_center < end_angle)
            # Combine the elliptical ring mask and slice mask
            combined_mask = ring_mask & slice_mask
            # Assign a unique value to each region for visualization
            roi_mask[roi_index,combined_mask] = 1
            roi_index += 1
            
    return roi_mask 

def create_elliptical_mask(arr,sigmax,sigmay,num_rings=5,num_slices=10):
    """
    Creates ROI masks of ellipses of equal probability rings
    
    Inputs:
    arr: Cropped detector array (nframes, height, width)
    sigmax: Standard deviation in the x-direction in pixels from Gaussian fit
    sigmay: Standard deviation in the y-direction in pixels from Gaussian fit
    
    
    """
    # Parameters
    _, height, width = arr.shape  # Image dimensions
    
    center = (height // 2, width // 2)  # Center of the image
    
    #Addresses strange issue of 1 being the last number or not
    #Outer boundary for rings of equal probabilities in fractions of standard deviations
    probs = np.arange(1/num_rings,1,1/num_rings)
    if probs[-1]>0.99 and probs[-1]<=1:
        probs[-1] = 0.998  
    else:
        probs = np.append(np.arange(1/num_rings,1,1/num_rings),0.998)
        
    
    #ring_widths = 1/np.sqrt(probs)
    ring_widths = np.sqrt(2)*erfinv(probs)
    print(ring_widths)
    
        
    # Create a grid of coordinates
    y, x = np.ogrid[:height, :width]

    # Calculate the elliptical distance from the center
    elliptical_distance = np.sqrt((x - center[1])**2/(sigmax)**2 + (y - center[0])**2/(sigmay)**2)
    
    #print(np.max(elliptical_distance))

    # Calculate the angle of each pixel relative to the center
    angle_from_center = np.arctan2(y - center[0], x - center[1])  # Angle in radians
    angle_from_center = (angle_from_center + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2*pi)

    # Define the boundaries for the elliptical rings
    #ring_width = 1.0 / num_rings  # Each ring will occupy a fraction of the elliptical distance range
    roi_mask = np.zeros((num_rings*num_slices,height, width))  # Create a blank mask for all ROIs
    #roi_mask[:,:,:] = -1 #This is for processing later. -1 is an unphysical value, unlike 0
    
    # Assign each ROI a unique index for visualization
    roi_index = 0
    
    for i in range(num_rings):
        # Choose ellipse rings
        if i==0:
            inner_radius = 0
            outer_radius = ring_widths[i]
        else:
            inner_radius = ring_widths[i-1]
            outer_radius = ring_widths[i]
        
        # Create a mask for each elliptical ring
        ring_mask = (elliptical_distance >= inner_radius) & (elliptical_distance < outer_radius)

        for j in range(num_slices):
            # Define angular boundaries for each segment
            start_angle = j * (2 * np.pi / num_slices)
            end_angle = (j + 1) * (2 * np.pi / num_slices)
            # Create a mask for each angular segment
            slice_mask = (angle_from_center >= start_angle) & (angle_from_center < end_angle)
            # Combine the elliptical ring mask and slice mask
            combined_mask = ring_mask & slice_mask
            # Assign a unique value to each region for visualization
            roi_mask[roi_index,combined_mask] = 1
            roi_index += 1
            
    return roi_mask 

    

### Creates square grid masks ###

def create_square_mask(arr, block_shape, mask_shape):
    l, m, n = arr.shape
    dy, dx = block_shape
    ny, nx = mask_shape

    yc, xc = m // 2, n // 2 #Image center
    y0, x0 = yc-(ny//2)*dy, xc-(nx//2)*dx #Loop starting point
    y, x = y0, x0 #Position variables
    
    blocks = []
    coords = []
    dPixels = [] #Number of Pixel distance in [x, y]
    
    for i in range(0, ny, 1):
        x = x0
        for j in range(0, nx, 1):
            blocks.append(arr[:,(y-dy//2):(y+dy//2), (x-dx//2):(x+dx//2)])
            coords.append((x, y)) 
            dPixels
            x += dx
        y += dy
    return blocks, coords

### For Converting from Real Space to Reciprical Space ###

def q_to_tth(q,lambDuh):
    return 2*np.arcsin(q*lambDuh/(4*np.pi))*(180/np.pi)

def reciprocal_space_map(lambDuh,tth,hor0,ver0,pix,sam2det,detShape):
    # Written by Vanya-GPT
    #lambDuh: beam wavelength [Å]
    #tth: detector angle posistion
    #hor0: horizontal beam position
    #ver0: vertical beam position
    #pix: pixel size [mm]
    #sam2det: sample to detector distance [mm]
    #detShape: 2x2 horizontal, vertical
    
    HorScattAngle=tth-np.arctan((np.arange(detShape[0,0],detShape[0,1]+1,1)-hor0)*pix/sam2det)*(180/np.pi); #In degrees
    VerScattAngle=-1*np.arctan((np.arange(detShape[1,0],detShape[1,1]+1,1)-ver0)*pix/sam2det)*(180/np.pi); #In degrees
    
    qx=(4*np.pi/lambDuh)*np.sin((HorScattAngle/2)*np.pi/180); # [Å^-1]
    qy=(4*np.pi/lambDuh)*np.sin((VerScattAngle/2)*np.pi/180); # [Å^-1]
    
    return qx,qy

def reciprocal_space_mapping(detector_width, detector_height, sample_to_detector_distance, pixel_size, two_theta, wavelength, initial_beam_position):
    # Convert two theta to radians
    two_theta_rad = np.deg2rad(two_theta)

    # Create a meshgrid for the detector pixels
    x = np.arange(detector_width)
    y = np.arange(detector_height)
    X, Y = np.meshgrid(x, y)

    # Calculate the pixel positions with respect to the initial beam position
    X = (X - initial_beam_position[0]) * pixel_size
    Y = (Y - initial_beam_position[1]) * pixel_size

    # Calculate the distance from each pixel to the sample
    Z = np.sqrt(sample_to_detector_distance**2 + X**2 + Y**2)

    # Calculate the angles of each pixel
    theta_x = np.arctan(X / sample_to_detector_distance)
    theta_y = np.arctan(Y / sample_to_detector_distance)

    # Calculate the reciprocal space coordinates
    k = 2 * np.pi / wavelength
    q_x = k * (np.cos(theta_x) - np.cos(two_theta_rad))
    q_y = k * np.sin(theta_y)
    q_z = k * (np.sin(theta_x) + np.sin(two_theta_rad))

    return q_x, q_y, q_z

### For Plotting ####

def make_detector_movie(imgs, scan_var,period, fig, ax, filename, fps,scale='log',clims = [0,10]):
    # Makes an animation of the detector images stored in `imgs`.
    # `fig` and `ax` are the Figure and Axes objects used to plot each movie frame
    # `filename` is the name of the output .gif file
    # `clims` is the colormap range
    # `fps` is the frame rate (frames per second) of the movie
    Nt = imgs.shape[0]
    
    if scale=='log':
        im = ax.imshow(imgs[0,:,:]+1, cmap='nipy_spectral',norm=LogNorm(vmin=-10, vmax=10))
    if scale=='gray':
        im = ax.imshow(imgs[0,:,:],cmap = 'gray',clim = (clims[0],clims[1]))
    else:
        im = ax.imshow(imgs[0,:,:], cmap='nipy_spectral',clim = (clims[0],clims[1]))
    
    
    def func(ii):
        im.set_data(imgs[ii,:,:])
        ax.set_title('Time = ' + str(np.around(period*scan_var[ii-1],2)) + ' seconds')
        return im
    
    anim = FuncAnimation(fig, func, frames=range(1,Nt))
    anim.save(filename + '.gif', writer=PillowWriter(fps=fps))

def create_color_list(length):
    # Get the default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Cycle through the colors and create a list of the specified length
    color_list = [color_cycle[i % len(color_cycle)] for i in range(length)]
    return color_list


### For Opening h5 Files ###

def print_h5_item(item, indent=''):
    """
    Recursively print the contents of an h5py group or dataset.
    
    Args:
    - item: The h5py group or dataset to print.
    - indent: A string of spaces used to indent nested items for better readability.
    """
    
    if isinstance(item, h5py.Group):  # Check if item is a group
        for key, subitem in item.items():
            print(f"{indent}/{key}")  # Print group name
            print_h5_item(subitem, indent + '    ')  # Recursively print contents of the group with additional indentation
    elif isinstance(item, h5py.Dataset):  # Check if item is a dataset
        print(f"{indent}[Dataset] Shape: {item.shape}, Type: {item.dtype}")
        # To print actual data, uncomment the line below. Be cautious with large datasets.
        # print(item[:])

        
### For Opening Batchinfo Files ###

def load_batchinfo(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line by the first occurrence of the colon
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert value to appropriate type (int, float, or leave as string)
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.startswith("[") and value.endswith("]"):
                        value = eval(value)
                    
            parameters[key] = value
    return parameters