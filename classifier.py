
# coding: utf-8

# In[1]:


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

from utils import extract_features_many_filenames

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split


# In[42]:


# To save as normal python script (easier to git diff)
get_ipython().system('jupyter nbconvert --to script classifier.ipynb')


# In[2]:


training_folders = {
    'vehicles': [
        'KITTI_extracted',
        'GTI_Right',
        'GTI_MiddleClose',
        'GTI_Left',
        'GTI_Far'
    ],
    'non-vehicles': [
        'GTI',
        'Extras'
    ]
}

training_filenames = {
    'vehicles': [],
    'non-vehicles': []
}

for folder, subfolders in training_folders.items():
    for subfolder in subfolders:
        files = glob.glob(folder + '/' + subfolder + '/' + '*.png')
        training_filenames[folder].extend(files)


# In[3]:


# Prepare training data
    # Load image
    # Extract features from image
    # Add features to training set
    # Preprocess data (might not be here?)


# In[4]:


# Smaller sample size while preparing pipeline
# sample_size = 500
# training_filenames['vehicles'] = training_filenames['vehicles'][0:sample_size]
# training_filenames['non-vehicles'] = training_filenames['non-vehicles'][0:sample_size]


# In[5]:


# PARAMETERS
config = {
    'color_space': 'LUV',      # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

    'spatial_feat': True,      # Spatial features on or off
    'hist_feat': True,         # Histogram features on or off
    'hog_feat': True,          # HOG features on or off
    
    # HOG features
    'orient': 9,               # HOG orientations
    'pix_per_cell': 8,         # HOG pixels per cell
    'cell_per_block': 2,       # HOG cells per block
    'hog_channel': 'ALL',      # Can be 0, 1, 2, or "ALL"
    
    # Spatial features
    'spatial_size': (16, 16),  # Spatial binning dimensions

    # Color hist features
    'hist_bins': 16,           # Number of histogram bins (for color histogram feature)
    'hist_range': (0, 256)     # Range for color histogram
}


# In[6]:


get_ipython().run_cell_magic('time', '', "# Feature extraction and preparation\ncar_features = extract_features_many_filenames(training_filenames['vehicles'], **config)\nnotcar_features = extract_features_many_filenames(training_filenames['non-vehicles'], **config)\n\nX = np.vstack((car_features, notcar_features)).astype(np.float64)                        \n# Fit a per-column scaler\nX_scaler = StandardScaler().fit(X)\n# Apply the scaler to X\nscaled_X = X_scaler.transform(X)\n\n# Define the labels vector\ny = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))\n\n# Split up data into randomized training and test sets\nX_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)\n\nprint('Feature vector length:', len(X_train[0]))")


# In[7]:


get_ipython().run_cell_magic('time', '', "# Create the classifier\nsvc = LinearSVC()\n\n# Train the classifier\nsvc.fit(X_train, y_train)\n\n# Classifier accuracy\nprint('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))")


# In[8]:


# Save the classifier (and the config for training so feature preparation
# is the same for training and prediction when classifier is loaded)
pickle.dump(svc, open('svc_classifier.pkl', 'wb'))
pickle.dump(config, open('feature_config.pkl', 'wb'))
pickle.dump(X_scaler, open('x_scaler.pkl', 'wb'))

