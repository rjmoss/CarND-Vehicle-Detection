## Robert Moss Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/test_image_1.jpg
[image2]: ./output_images/test_image_2.jpg
[image3]: ./output_images/test_image_3.jpg
[image4]: ./output_images/test_image_4.jpg
[image5]: ./output_images/test_image_5.jpg
[image6]: ./output_images/test_image_6.jpg

[image7]: ./output_images/search_window_64.jpg
[image8]: ./output_images/search_window_96.jpg
[image9]: ./output_images/search_window_120.jpg
[image10]: ./output_images/search_window_160.jpg
[image11]: ./output_images/search_window_200.jpg
[image12]: ./output_images/search_window_260.jpg

[image13]: ./output_images/hogs.jpg
[image14]: ./output_images/All_positive_windows.jpg
[image15]: ./output_images/Heatmap.jpg
[image16]: ./output_images/Thresholded_Heatmap.jpg
[image17]: ./output_images/Labelled_array.jpg
[image18]: ./output_images/Result.jpg

[image19]: ./output_images/heatmap_frames.jpg
[image20]: ./output_images/combined_frames.jpg

[video1]: ./output_images/video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Structure
The project was structured as 2 IPython notebooks and 1 python file:
* `utils.py` contains the bulk of the code, including the feature extraction functions (HOG, spatial, colour) which are used by both of the notebooks
* `classifier.ipynb` contains the code for training the classifier (including saving the classifier, transformer and preprocessing configuration parameters)
* `pipeline.ipynb` contains the image/video processing pipeline
 
---
### Writeup / README

You're reading it!

### Classifier

#### Histogram of Oriented Gradients (HOG)

The code for extracting the HOG features from an image is in the function `get_hog_features` on line 46 of `utils.py`. I experimented with different `skimage.hog()` parameters and the parameters which were found to give the highest accuracy can be seen in `classifier.ipynb` cell 4, the HOG specific parameters being:

```
    'color_space': 'YUV',      # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

    'orient': 11,              # HOG orientations
    'pix_per_cell': 8,         # HOG pixels per cell
    'cell_per_block': 2,       # HOG cells per block
    'hog_channel': 'ALL',      # Can be 0, 1, 2, or "ALL"
```

Here are examples of the `vehicle` and `non-vehicle` classes with their corresponding HOG features:

![alt text][image13]


#### Spatial and colour features
In addition to the HOG features, a histogram of colours and binned spatial features were added to the feature array.

####  Classifier training
The steps below for data preprocessing and classifier training can be seen in cells 5-6 in `classifier.ipynb`.

Having extracted the features from the images, `StandardScaler().transform` from sklearn was used to normalise the data. The data was then separated randomly into a training set and testing set using `train_test_split` from sklearn with an 80:20 split between training and test data. An SVM classifier (`LinearSVC()`) was used to fit the data and make predictions and finally `sklearn.metrics.classification_report` was used to output the following accuracy report:

|           | precision | recall | f1-score | support |
| --------- |:---------:| ------:| --------:| -------:|
| non-cars  |  0.9944   | 0.9950 | 0.9947   | 1787    |
| cars      |  0.9949   | 0.9943 | 0.9946   | 1765    |
| avg/total |  0.9947   | 0.9947 | 0.9947   | 3552    |

As can be seen from the above the final accuracy was 0.9947 with similar accuracy on car/non-car images (which shows the model has little bias between the classes).


### Pipeline
Below is the pipeline for images, the same core pipeline was used for each frame of the video except that the heatmap was constructed from previous frames (see discussion below).

#### Sliding Window Search
The window sizes chosen can be seen in the 6th cell of `classifier.ipynb`. Five windows from 64px to 260px were chosen to cover the full range of car positions from far to close.

The area which the windows can search in was modified depending on the size of the windows, this is because smaller windows are intended to pick up cars near the horizon (centre of the image) whereas larger windows should pick up closer cars. Here are a couple of examples of window size and search area:

![alt text][image7]
![alt text][image10]

The overlap needs to be large enough to not accidentally jump over a car without identifying it however if too high the overlap would slow the pipeline and be unnecessary, and just identify the same vehicle repeatedly. An overlap of 75% was chosen mostly through experimentation of different sizes.

The sliding window search was originally implemented in the function `slide_window` and `search_windows` on lines 143/190 respectively of `utils.py` to get the windows and loop through them. However this is slow as it involves extracting the HOG features for every window, alternatively the function `find_cars` on line 230 extracts the HOG features once and then subsamples this feature array for each window.

#### Heatmap
A heatmap of the positive windows is created by looping through the windows and adding an amount of "heat" (defaulted to 1) to an array of zeros (see `add_heat` on line 299-312 in `utils.py`). In addition extra heat (see `add_extra_heat` on line 305) is added to boxes which overlap other boxes (this helps remove false positives and also helps get the full vehicle when thresholding rather than just the intersection of the positive windows).

The heatmap is *thresholded*, meaning that values below a certain threshold are discarded. This helps remove false positives. The threshold was chosen as 4, this was done through experimenting with a variety of values (running against the whole video - see below).

Next `scipy.ndimage.measurements.label()` was used to label the heatmaps, this applies a different label to each non-zero area of the thresholded heatmap.

![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

Finally we draw the bounding box as the extremity of the labelled regions (see `draw_labelled_bboxes` on line 213 of `utils.py`).

![alt text][image18]


#### Test images
Here are some examples of the performace of the pipeline on the test images:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

---

### Video Implementation

#### Final video
Here's a [link to my video result](./output_images/video_result.mp4)

#### Heatmap
The heatmap for each individual frame is calculated as for images above, however there is one extra step before applying the threshold to the heatmap: the pre-threshold heatmap is made as the average heatmap over the previous 8 frames, this helps to reduce false positives as we assume an actual car doesn't move much between frames. Therefore if there is a window in just one of the last 8 frames it is likely to be a one off false positive, and the averaging before thresholding will remove it from the final heatmap.

Note that an average is used rather than a sum so that the same threshold can be used for images and clips, and the first few clips of a frame still work.

Here is an example over just 4 frames:

![alt text][image19]

You can see from the images above that there are a couple of false positives however the true positives are much stronger on the heatmap. Averaging and then thresholding the above 4 heatmaps gives the following heatmap and bounding boxes with the false positives removed:

![alt text][image20]

---

### Discussion

While working on this project I had particular issues identifying the white car when it was either far away or very close (in between the identification was very successful). I mostly fixed these issues by choosing appropriate window sizes and 

Another issue was the slow pipeline time, which made it hard to test. One thing which helped was implementing the variable y start-stop based on window size, so I didn't waste time searching the wrong bit of the image. This gave a speed increase of about 50%. A future improvement could be to use opencv hog as this is quicker.

The threshold selection was a challenge, but I was able to test multiple different thresholds at a time by colouring the bounding boxes differently according to the threshold value they were based on. This sped up development time considerably.

I was pleased with the SVM accuracy of >99% however more training data, or paramter tweaking, could lead to an even higher accuracy which could decrease the number of false positives.
