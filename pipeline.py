
# coding: utf-8

# In[1]:


# Imports
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

from utils import slide_window, extract_features_single, draw_boxes, find_cars


# In[3]:


# To save as normal python script (easier to git diff)
get_ipython().system('jupyter nbconvert --to script pipeline.ipynb')


# In[2]:


# Load test images (RGB)
test_image_paths = glob.glob('test_images/test*.jpg')
test_images = [mpimg.imread(file) for file in test_image_paths]


# In[3]:


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, config):
    on_windows = []

    for window in windows:
        window_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        
        features = extract_features_single(window_img, **config)

        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)

        if prediction == 1:
            on_windows.append(window)

    return on_windows


# In[4]:


svc = pickle.load(open('svc_classifier.pkl', 'rb'))
config = pickle.load(open('feature_config.pkl', 'rb'))
X_scaler = pickle.load(open('x_scaler.pkl', 'rb'))


# In[12]:


test_img = test_images[0]
draw_image = np.copy(test_img)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255

windows = slide_window(test_img,
                       x_start_stop=[None, None],
                       y_start_stop=[400, 700],
                       xy_window=(96, 96),
                       xy_overlap=(0.5, 0.5))

hot_windows = search_windows(test_img, windows, svc, X_scaler, config)                       

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    


plt.imshow(test_img)
plt.figure()
plt.imshow(window_img)
plt.show()


# In[ ]:


find_cars(test_img, ystart=400, ystop=700, scale=1, svc=svc, X_scaler=X_sca, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins, color_space)

