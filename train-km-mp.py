###############################################################################
#                 KM-GEN General Purpose Image Classifier  
#                      Sensor Analytics Australia™ 2024
###############################################################################

import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
from sautils3_5 import num_name,calcEntropy,writeLog,imgcont,save_list,\
     chunk,workpacks,mkdir_cleared,color,check_img,imgFeats,fsearch,\
     imgResize_n,fileTs,invarPR,imgbw,hu_invars
            
from config import ImgPath,cSz,wdir,deBug,kiter,kmethod,ninit,nfts,imgfull,\
     imght,saverej,imgdist,rseed,img_bw,ktol,c_old
import matplotlib
import os
import cv2
from datetime import datetime
from multiprocessing import Pool
import glob,tqdm

def p_img_read(name):
 p_img=cv2.imread(name)
 return p_img
def img_load_proc(workpacket):
 tid = workpacket['id']
 fpath = workpacket['imgP']
 flist = workpacket['images']
 file_fnames  = workpacket['outPath_fnames']
 file_data  = workpacket['outPath_data']
 data = []
 fnames = []
 kn=knt=knf=0
 if deBug == 2:
    print('Task id:',tid)
 for img in flist:
     if len(sys.argv) == 6:
        fDts = fileTs(img)
        if fDts < int(st_d_t) or fDts > int(en_d_t): continue #skip this file
     fDt=num_name(img)
     if fDt != -1 :
        imgf=os.path.join(fpath,img)
        if check_img(imgf):
           imgd=p_img_read(imgf)
           ifeats = imgFeats(imgd,nfts) # find n specified img features
        else: 
           print('unnumbered file found {}', format(img))
           knt +=1
           continue
        # Assemble Features Vector
        data_vec = []
        if imgfull: # use full image for analysis
           imgRsz = imgResize_n(imgd,imght)
        if imgdist == 0 or imgdist == 1: # check image for correct features 
           if ifeats is None:
              if deBug:
                print('No features found! skipping:{}...'.format(img))
              knt +=1
              continue
           if len(ifeats) < nfts:
              if deBug:
                print('{} not enough features found skipping:{}...'
                                  .format(ifeats.shape[0],img))
              knt +=1
              continue
           if len(ifeats) > nfts:
              if deBug:
                print('{} TOO MANY FEATURES! Increase nfts! skipping:{}...'
                                  .format(ifeats.shape[0],img))
              knt +=1
              continue
        if imgdist == 2: # hu invars don't use nfts, only for img check
           if ifeats is None or len(ifeats) < nfts: 
              if deBug:
                print('none or insufficient features found skipping:{}'
                                   .format(img))
              knt +=1
              continue
        # Features: img_gs_flattened or img_des_flattened
        if imgfull:
           if len(imgRsz.shape) > 2:
                imgRsz=cv2.cvtColor(imgRsz, cv2.COLOR_BGR2GRAY)
           if img_bw: _,imgRsz=imgbw(imgRsz) # use pure black and white image
           data_vec = imgRsz.flatten()
        elif imgdist == 1: # invoke orb descriptors
           data_vec,hd = ipr.desDists(imgd)
           if deBug:
              if hd == 0 :
                 print(ipr.imgid,'->',imgf,'diff: ',hd)
           if len(data_vec) != nfts:
              print('ERROR!!!',ipr.orb.shape,imgf)# sanity check
              sys.exit(1)
        elif imgdist == 2: # invoke Hu's moment invariants
           if img_bw: data_vec = hu_invars(imgbw(imgd)[1])
           else: data_vec = hu_invars(imgd)
        else:
           data_vec = ifeats.flatten()
           if len(data_vec) != nfts*4: 
              print('ERROR!!!',ifeats.shape,imgf)# sanity check
              sys.exit(1)
        data.append(data_vec)
        fnames.append(img)
        kn +=1
     knt +=1
 if deBug == 2:
     print('proc:{} Total images processed: {} selected: {} '
           .format(tid,knt,kn))
 with open(file_fnames,'wb') as fpkl:
     fpkl.write(pickle.dumps(fnames))
 with open(file_data,'wb') as fpkl:
     fpkl.write(pickle.dumps(data))
 wDir=os.path.dirname(file_fnames)
 ffstats=os.path.join(wDir,'proc_{}_stats.pkl'.format(tid))
 with open(ffstats,'wb') as fpkl:
     fpkl.write(pickle.dumps('{},{}'.format(knt,kn)))
 return None

##### Main #####
if __name__ == "__main__":

 if len(sys.argv) != 4 and len(sys.argv) != 6:
     print('USAGE: on|off  0|1 numClusters [YYYYMMDDHHMMSS YYYYMMDDHHMMSS]')
     print('on: show progress bar off: turn off for cron etc')
     print('0:Elbow Analyses 1:Actual KMeans')
     print('optional start and end dates for YYYYMMDD-HHMMSS dated filenames')
     sys.exit(1)

 if not os.path.exists(ImgPath):
     print('fix image path in this code, it  does not exist!',ImgPath)
     sys.exit(1)
 mname = 'km_model.pkl' # for trained km model saved as message
 print('total images in {} : {}'.format(ImgPath,len(os.listdir(ImgPath))))

 ##########  Data Input Block #################
 pbar = sys.argv[1] # progress bar 'on' or 'off' 
 opt = int(sys.argv[2]) # 0 for elbow analyses 1 for actual km clustering
 nC = int(sys.argv[3]) # number of clusters for elbow analyses or training 
 if len(sys.argv) == 6:
    st_d_t=datetime.strptime(sys.argv[4],'%Y%m%d%H%M%S').timestamp()
    en_d_t=datetime.strptime(sys.argv[5],'%Y%m%d%H%M%S').timestamp()
    if len(sys.argv[4]) !=14 or len(sys.argv[5]) !=14:
       print('input datetime error')
       sys.exit(1)
    if int(st_d_t) > int(en_d_t):
       print('input datetime error - inconsistent dates')
       sys.exit(1)
 ############### MultiProcessing Block ###############################
 if c_old: # re-use training files from last run with same ImgPath
  dataNormed = pickle.load(open('dataNormed.pkl','rb'))
  kntvars = pickle.load(open('vars.pkl','rb'))
  knt_tot = kntvars[0] 
  knt_tot_sel = kntvars[1] 
  ImgPath_old = kntvars[2]
  ImgPath_tot = len([name for name in os.listdir(ImgPath) 
                   if os.path.isfile(os.path.join(ImgPath, name))])
  if knt_tot > ImgPath_tot or ImgPath != ImgPath_old : # sanity check
      print(color.RED+'ERROR {} Images:{} != total images in model:{}'
             .format(ImgPath,ImgPath_tot,knt_tot))
      print('re-use data outdated! quitting'+color.END)
      sys.exit(1)
  print('Re-using training data saved from the last run')
 else:
  if imgdist == 0 and not imgfull:
      print(color.BLUE+'*ORB keypoints PR Started*'+color.END)
  if imgdist == 1 and not imgfull: 
      ipr = invarPR(ImgPath,nfts) # for hamming dist metric use
      imgr1 = ipr.imgrand
      print(color.YELLOW+'*ORB desc. invariant PR started, index: {} rseed: {}*'
            .format(imgr1,rseed))
      print(color.END)
  if imgfull: 
      print(color.PURPLE+'*Full Image PR Started*'+color.END)
  if imgdist == 2 and not imgfull:
      print(color.CYAN+'*Hu invariant PR Started*'+color.END)
  if img_bw: print(color.DARKCYAN+'*black and white images used*'+color.END)

  mkdir_cleared(wdir) # working dir to save serialised mproc outputs
  img_ls=os.listdir(ImgPath)
  img_ls_chunk=list(chunk(img_ls,cSz)) 
  workpckts=workpacks(ImgPath,img_ls_chunk,workDir=wdir)

  pool = Pool()
  if pbar == 'off':
      tRun=True
      print('Progress Bar:',pbar)
  else: 
      tRun=False
      print('Progress Bar',pbar)
  for _ in tqdm.tqdm(pool.imap_unordered(img_load_proc,workpckts),
                    total=len(workpckts),colour='magenta',
                    disable=tRun,leave=True):
    pass
  pool.close()
  pool.join()

  #################### Integrate mp output files in working dir #######
  print('integrating output files in {} ...'.format(wdir))
  fnames_all = glob.glob(wdir+'/*fnames*.pkl')
  data_all = glob.glob(wdir+'/*data*.pkl')
  stats_all= glob.glob(wdir+'/*stats*.pkl')

  fnames = []
  data = []
  for i,j in zip(fnames_all,data_all):
     fnames.extend(pickle.load(open(i,'rb')))
     data.extend(pickle.load(open(j,'rb')))
  knt_tot=knt_tot_sel=0
  for f in stats_all:
     s = pickle.load(open(f,'rb'))
     ss=s.split(',') 
     knt_tot += int(ss[0])
     knt_tot_sel  += int(ss[1])
  print('Total images:{} Images found:{}'
        .format(knt_tot,knt_tot_sel))
  if knt_tot_sel == 0: sys.exit(1) 

  ############# Saving Outputs ################################
  print('saving outputs')
  save_list(fnames,'fnames.txt')
  data = np.array(data)
  np.set_printoptions(suppress=True, formatter={'float_kind':'{:.1f}'.format})
  datamax = data.max(axis=0) #later for sanity checking 
  scaler = preprocessing.StandardScaler().fit(data)
  dataNormed = scaler.transform(data) 
  np.savetxt('datamax.txt',datamax, fmt='%.1f') # datamax saved for this run
  np.savetxt('fnames.txt',fnames,delimiter=" ",fmt="%s") # fnames also saved 
  print('*fnames.txt saved for prediction module*')
  np.savetxt('data.txt',data,fmt='%.1f') # to inspect original training data
  with open('dataNormed.pkl','wb') as fpkl: # data set saved in pickle format
      fpkl.write(pickle.dumps(dataNormed))
  print('*dataNormed.pkl saved for prediction module or re-use*')

  # Sanity Checks
  if knt_tot_sel < 1:
     print('No Images found exiting!')
     sys.exit(1)
  if knt_tot_sel < nC:
     print('Images found < number of cluster (nC):{} decreasing nC'
            .format(nC))
     nC = knt_tot_sel
  kntvars = [knt_tot,knt_tot_sel,ImgPath]
  with open('vars.pkl','wb') as fpkl: # also save these variables re-use
      fpkl.write(pickle.dumps(kntvars))
  print('*vars.pkl saved for re-use*')

  # end if c_old

 ######### Elbow Analyses and Display Block ###############
 if opt == 0:
    matplotlib.use('TkAgg') # to avoid cv2 qt conflict
    inertias = []
 
    print('Getting ready to display')
    for i in range(1,nC+1):
       print('.',end='')
       km = KMeans(init=kmethod,n_clusters=i,n_init=ninit,max_iter=kiter,\
                  tol=ktol)
       km.fit(dataNormed)
       inertias.append(km.inertia_)
       print('clusts:{} intertia:{} iters:{}'.format(i,km.inertia_,km.n_iter_))
    print('\n')
 
    print('press [q] inside the display chart to end')
    plt.plot(range(1,nC+1), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show(block=True) 
    sys.exit(0)
 ################### KMeans Block ################################
 print('ready to train KMeans with:',nC,'clusters')
 km = KMeans(init=kmethod,n_clusters=nC,n_init=ninit,max_iter=kiter,\
             tol=ktol) 
 km.fit(dataNormed)
 pickle.dump(km, open(mname, 'wb')) # dump trained model
 print('*KMeans trained model saved as:{} for prediction module*'.format(mname))
 ################## Save rejected filenames ######################
 if knt_tot_sel < knt_tot and saverej == 'yes':
    print('saving rejected filenames')
    if os.path.exists('ffnames-rej.txt'):
       os.remove('ffnames-rej.txt')
    fcnt=0
    for x in os.listdir(ImgPath):
       if fsearch('fnames.txt',x) == -1: # i.e. not found
         with open('ffnames-rej.txt','+a') as f:
           f.write(os.path.join(ImgPath,x)+'\n') 
         fcnt += 1
    if fcnt > 0 : 
       print('*{}% rejected frames in ffnames-rej.txt deBug=1 for info*'
                         .format(round(fcnt/knt_tot*100))) 
 print('* ...training done {}% images rejected km intertia:{} & iters:{} *'.
       format(round((knt_tot-knt_tot_sel)/knt_tot*100),
              round(km.inertia_,2),km.n_iter_))
 if imgdist == 1 and not imgfull and not c_old: 
     imgr2 = ipr.imgrand
     if imgr1 != imgr2:
       print(color.YELLOW+'*invariant PR ended, new index: {} *'
           .format(ipr.imgrand))
     print(color.END)
