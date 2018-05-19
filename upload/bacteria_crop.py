import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

#finding the position from where the strip starts
def starting(arr):
    start = 0 
    for i in range(len(arr)):
        if (arr[i] == 1):
            start = i+1
            return(start)
     
#finding position where the strip ends
def ending(start, arr):
    end = 0
    for i in range(start, len(arr)):
        if (arr[i] == 0):
            end = i-2
            return(end)
  
#creating cropped image using the start and end points
def cropped(path,arr):
    
    #finding initial end points of strip
    start = starting(arr)
    end = ending(start,arr)
    size = img.shape[1]
    
    #cropping based on the found points
    img1 = Image.open(path)
    img2 = img1.crop((0, start, size, end))
    img2.save("cropp.jpg")
    plt.imshow(img2)
    
    return(img2)
        
    
#importing the image
path = 'path of image with the image name'
img = cv2.imread(path)

#converting image to grayscale in order to apply Kmeans
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#plt.imshow(imgray)

X = np.array(imgray)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#print("\nCluster Centers:",kmeans.cluster_centers_)

Label = (kmeans.labels_)
#print(Label)

#calling function to crop the image
img2 = cropped(path,Label)



