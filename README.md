
### About ###
`KM-GEN` is an unsupervised classifier for finding groups of similar or dissimilar images (anomalies) in large collections. It's used for analysing image repositories e.g. from auto-captured video footage e.g. travel VLOGs, driving tours, GoPro/dash-cam videos, security cam snaps/videos etc. Commonly occurring frames can be filtered out leaving only images which are relatively unique and thus may be of interest. A long video may be converted into a short time-lapse video of highlights or a large image collection can be condensed into a slide show of relatively unique images.

### Demos ###
Image classification and video processing examples.

#### Example Image Classification
Six images of two different type of flowers classified into two clusters using `imgdist=3` for Hu's moment invariants with colour support. 
```
python train-km-mp.py on 1 2
python predict-km.py i -1
```
Option *-1* in `predict-km.py` allows selecting specific clusters for finding out the images present in each cluster.

##### Cluster 0
![Tulips](./flowers/clust0.jpg)

##### Cluster 1
![Iris](./flowers/clust1.jpg)

#### Examples Video Analytics
Following clips were produced for comparing the three main `KM-GEN` methods.

##### Features vs Descriptors vs Invariants
[Vincent Price | The Last Man on Earth (1964) an A.I. Rendition](https://youtu.be/dgB6E9QJVKk?si=J3d7sTJOuMGVpvhK) is about an eight minutes `KM-GEN` compilation of atypical scenes from 86 minutes full movie with 100 *ORB* keypoint features.

[Vincent Price | The Last Man on Earth (1964) an A.I. Rendition (Color)](https://youtu.be/mgjJMnhwmpM) is about seven minutes `KM-GEN` compilation of atypical scenes from 86 minutes full movie with 100 *ORB* descriptors using the Hamming Distances approach; *invariant pattern-recognition* section.

[Vincent Price | The Last Man on Earth (1964) an A.I. Rendition (Color)](https://youtu.be/xaTTK8JK-Gs) is about six minutes `KM-GEN` compilation of atypical scenes from 86 minutes full movie with Hu's moment invariants.

##### Dashcam Video Scanning
[Dash Cam Tours | 3-Hours Dash Cam Video Scan with A.I.](https://youtu.be/g8BbILWP7_8) is a 1.2 minute `KM-GEN` scan of the three hours long dash cam video with Hu's moment invariant (00:16 time-stamp 2442.873 aircraft crossing above the freeway).

Hu's method is the fastest. It selects more accurately, collecting more novel scenes at some loss of continuity between the scenes making the time-lapse a bit choppier.

[Swedish Railcam | High Speed Polar Train an A.I. Rendition](https://youtu.be/rNEWurCoHp8?si=DAtGZgXiaU_yWRPr) runs a 3 hours 41 minutes train cabin cam from Narvik-Pitkäjärvi in under one minute. *ORB* descriptors' option used. [Hu Invariant version](https://youtu.be/jUA5yO4TNNw) is added for comparison.

### Usage
* The images should be of adequate resolution, e.g. 480 x 640 or above.
* The images have adequate features such as in street scenes, landscapes, objects, people etc. For instance trying to analyse tiny MNIST images or very dark scenes will not work as these are of extremely low resolution/contrast and thus not amenable for feature analysis. Feature analysis can however be replaced with full image analysis by enabling `imgfull` and `img_bw` options in `config.py`; reducing `imght` can benefit images with scant details. 
* The `imgfull` option should be enabled for low-light frames such as those captured in night vision mode of security cameras or `nfts` value can be lowered to double digits in this case. Alternatively `KM-MOD` may be used which is designed specifically for security cameras.

### Underlying Principle
The algorithm classifies images into clusters using KMeans. When the number of clusters is close to optimal, we will find clusters within 1st (25%) Quartile containing interesting images. 
**NB:** `train-km-mp.py` option *0* enables *Elbow Analysis*, which is a good measure of finding the optimal number of cluster for the data set.

### Requirements
* RPI5 with 8GB is highly recommended however RPI4B with 4GB should be adequate in most cases.
* Python 3.7.3 or higher

### Dependencies
```
sudo apt update
sudo apt upgrade
sudo apt install ffmpeg

python -m pip install -U pip
python -m pip install -U scikit-image 
pip install opencv-python
pip install shutils
pip install -U scikit-learn
pip install matplotlib
pip install tqdm
pip install yt-dlp
```

### Config
* `ImgPath` to `my_output_folder`, or whatever you may have named it, needs to be edited in `config.py`. Other parameters can be left as is for time being.
* Set the *Path Variables* at start of `moviefrm-list`, `moviefrm-list-ni`, and `utils/done-driver-mp` bash scripts to the actual paths on your computer. 
**NB** The variable `DV` value in `utils/daily-driver-mp` and `utils/date-driver-mp` must be exactly the same as in `moviefrm-list-ni` if using these scripts. Also the paths have to edited as above.

### Example
Clone this repository then extract frames from any MP4 movie clip (not included):
```
git clone https://github.com/SensorAnalyticsAus/KM-GEN.git
cd KM-GEN
./utils/fextract my_travel_vlog.mp4 my_output_folder 1
```

#### Step 1 train
`$ /path/to/.venv/bin/python train-km-mp.py on 1 10`. Where *on* shows the progress bar, *1* to run in normal mode, and *10* is the number of clusters to use for training on the images, usually this a good number to start with e.g. youtube videos, however more precise value should be obtained by using option *0*.

#### Step 2 predict (output frames from selected clusters from step 1)
`$ /path/to/.venv/bin/python predict-km.py ni 25`. The predict module will run in non-interactive mode with *ni* option and gather up cluster of images less than or equal to 25 percentile.

#### Step 3 create a time-lapse video
Edit `moviefrm-list` shell script and change the following variables to your own values:
```
DIRP=/mnt/SSD
DV=YT
```
`$ ./moviefrm-list 1 ffnames.txt`. This will create a time-lapse video of the selected frames in Step 2 and display these at 1 frame/sec.

### Invariant Pattern Recognition
Invariant methods are not overly affected by the images being rotated.
Setting `imgdist > 0` enables invariant pattern recognition methods such as *ORB* descriptors and Hu's moment invariants being used instead of keypoint features. Generally Euclidean distance is used however for *ORB* descriptors, an index frame is randomly chosen and the Hamming Distances of all other frames are calculated with reference to this frame. 

### Config PR Options
The following `imgdist` values select different PR algorithms with the exception of `imgfull` option.

* 0: *ORB* keypoint features
* 1: *ORB* descriptors
* 2: Hu moment invariants on grayscale images
* 3: Hu moment invariants with RGB support
* 4: Colour histograms
* 5: Image's upper left corner's data from discrete Cosine Transformation  
* 6: Eigen values of single objects against uniform background (as in Eigenfaces) 
* 7: Image contours and entropy for motion-detection in security camera frames
* `img_bw` flag for converting images to black and white is accepted for `imgfull` and `imgdist = 0,1,2,3` options.
**NB:** Enabling `imgfull` over rides all the above options. 

### Utils
* `./utils/done-driver-mp` accepts `-h` to display usage information. This is a general purpose utility, which runs in batch mode with user specified parameters, to create a time-lapse video of all images in a folder.

* `./utils/fextract` accepts `-h` to display usage information. This utility is for extracting images from videos. It provides optional parameters `[skip_no_ts|simple_no_ts]` for extracting frames without the default timestamps (in secs) by skipping non-key frames or using the default `ffmpeg` mode. 

* `./utils/save-km`  *usage: {filename}*. Utility to save trained KMeans model for re-use in `train-km-mp.py` or `predict-km.py`.`ImgPath` must point to the same images folder with which the model was trained with.

* `./utils/daily-driver-mp` accepts `on|off` to display progress bar or run in silent mode (e.g. for use in cron). This utility is for security cam images with filenames in [OCD3](https://github.com/SensorAnalyticsAus/OCD---OpenCv-motion-Detector) or Foscam date-time format (e.g. `img_20240515-223903_019269.jpg`. It runs in batch mode collecting all images from time now till 12 hours in the past for a time-lapse summary of events. Recommended `imgdist=3`.
* `./utils/date-driver-mp` accepts `-h` to display usage information. This utility is also for security cam images. It converts images from user specified date-time range into a time-lapse video. Recommended `imgdist=3`

### Troubleshooting
* An incorrect path being set in `config.sys` or the bash scripts.
* Too few images being selected. Either `nfts` can be progressively lowered towards a minimum of *3* or `imgfull` analysis option may be invoked.
* Images are in an unrecognised format, convert all such images to JPG.
* Images sizes differ. 
* Not getting good clustering with `imgdist=0|1`? Increase `nfts`. Note: increasing `nfts` does neither impact `imgfull=1`nor `imgdist > 1` options. 
