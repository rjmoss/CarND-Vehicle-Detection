import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg

CONVERT_RGB_TO = {
        'HSV': cv2.COLOR_RGB2HSV,
        'LUV': cv2.COLOR_RGB2LUV,
        'HLS': cv2.COLOR_RGB2HLS,
        'YUV': cv2.COLOR_RGB2YUV,
        'YCrCb': cv2.COLOR_RGB2YCrCb
}

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))


# Define a function to compute color histogram features
def color_hist2(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0], bins=32, range=(0, 256))
    ghist = np.histogram(img[:, :, 1], bins=32, range=(0, 256))
    bhist = np.histogram(img[:, :, 2], bins=32, range=(0, 256))
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features



def extract_features_single(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                 pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True,
                 hog_feat=True):

    if color_space != 'RGB':
        feature_image = cv2.cvtColor(img, CONVERT_RGB_TO[color_space])
    else:
        feature_image = np.copy(img)

    img_features = []
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], orient,
                                                     pix_per_cell, cell_per_block, vis=False,
                                                     feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell,
                                            cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)

    return np.concatenate(img_features)


def extract_features_many(imgs, *args, **kwargs):
    return [extract_features_single(img, *args, **kwargs) for img in imgs]


def extract_features_many_filenames(img_files, *args, **kwargs):
    all_features = []
    for file in img_files:
        img = mpimg.imread(file)
        img_features = extract_features_single(img, *args, **kwargs)
        all_features.append(img_features)

    return all_features


def read_img(filename, reader='mpimg'):
    """
    The training dataset provided for this project ( vehicle and non-vehicle images) are in the .png
    format. Somewhat confusingly, matplotlib image will read these in on a scale of 0 to 1, but
    cv2.imread() will scale them from 0 to 255. Be sure if you are switching between cv2.imread()
    and matplotlib image for reading images that you scale them appropriately! Otherwise your
    feature vectors can get screwed up.

    To add to the confusion, matplotlib image will read .jpg images in on a scale of 0 to 255 so if
    you are testing your pipeline on .jpg images remember to scale them accordingly. And if you take
    an image that is scaled from 0 to 1 and change color spaces using cv2.cvtColor() you'll get
    back an image scaled from 0 to 255. So just be sure to be consistent between your training data
    features and inference features!
    """
    # TODO - not done properly yet - should probably return range 255 as am converting colour later
    # anyway using cv2.cvtColor() (in fact does it even matter if I'm doing this later?)
    if reader == 'mping':
        img = mpimg.imread(filename)
    elif reader == 'cv2':
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError('Unrecognised reader')

    return img