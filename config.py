# KM-GEN CONFIG
#
c_old=0   # '0' compute new training data '1' re-use training data from lastrun
ImgPath='./flowers/data' # images path 
cSz=5                                       # n imagefiles/workpacket
wdir='./tmp-pkl'                            # path to working tmp dir
nfts=10    # features required 
ktol=1e-15 # default 1e-4
kiter=1000 # set max_iter value, default=300
kmethod='random' # random or k-means++
ninit=10   # a numeric value e.g. 10 or 'auto', later is non-deterministic
loadfit=0  # '0' use labels from training '1' recompute labels with fit_predict
saverej='no' # 'yes' or 'no' save or don't save rejected image file names
rseed=20 # any number for reproduceable indices for imgdist or 'off' to disable
deBug=0    # '0' quiet mode '1' feature forming messages '2' more info
imgfull=0  # '0' disables full image   | '1' uses full image overrides imgdist
imght=100  # adjust full image height to imght pixels same A/R for imgfull
img_bw=0   # '0' use default grayscale | '1' change to black and white
imgdist=3  # '0' use keypoint features | '1' use descriptors' hamming dists
           # '2' use Hu's invariants   | '3' use Hu's invariants with colour
           # '4' use colour histograms | '5' use dominant DCT values
           # '6' use for single objects| '7'motion-detect sec cams 
#
# Usage:
# Extract frames from video:
# $ mkdir images
# $ ffmpeg -i some_video.mp4 -r 1 images/img_%05d.jpg
# Edit config.py setting 'ImgPath' to images/ folder. For low resolution images
#      'imgfull' should be set to '1', where by the entire image gets analysed
#      instead of its main features. 'nfts' may be lowered to reduce frames
#      being rejected. NB: 'imgfull' over rides all other options.
# $ python train-km-mp.py on 1 8
# $ python predict-km.py ni 25
# $ ./moviefrm-list 10 ffnames.txt 
#
# About #
# This classifier works on image features, which makes it more accurate
# but a bit quirky as images with fewer or more than nfts features leads to 
# those being rejected. Generally this will not impact the results for 
# frames extracted from video files such as mp4.
