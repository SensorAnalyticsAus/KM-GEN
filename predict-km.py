###############################################################################
#                  KM-GEN General Purpose Image Classifier
#                      Sensor Analytics Australia™ 2024
###############################################################################

import sys,os,argparse
from config import ImgPath as prefixP
from config import loadfit,imgdist,imgfull,img_bw
from sautils3_6 import whereinC

CLI = argparse.ArgumentParser(epilog='Note for INTERACTIVE MODE: Enter perc = 0\
                for percentile option. Enter perc = -1 for\
                specific cluster option.')
CLI.add_argument('inter',help='{i|ni} interactive mode|non-interactive mode.')
CLI.add_argument('perc', help='{percentile_value|-ve_cluster_index}\
                  1st val selects clusters <= percentile (usually 20-80)\
                  2nd val selects a specific cluster by entering its cluster\
                  index value with a minus sign.') 
args = CLI.parse_args()

import numpy as np
from sklearn.cluster import KMeans
import pickle

if not os.path.exists(prefixP):
    print('path',prefixP,'does not exist! fix prefixP')
    sys.exit(1)

fnames = np.loadtxt('fnames.txt',dtype=str)
print('fnames loaded:',len(fnames),'lines')

km = pickle.load(open('km_model.pkl','rb')) 
cc = km.cluster_centers_
if loadfit:
    dataNormed = pickle.load(open('dataNormed.pkl','rb'))
    labels = km.fit_predict(dataNormed)
else: labels = km.labels_

nC = len(cc)
print('clusters in this model:',nC)

membersC = [] #members in each cluster
mC = []
for i in range(0,nC):
    membersC.append(np.where(labels == i)[0])
    mC.append(len(membersC[i]))
mCs = sorted(mC)
if int(args.perc) >= 0:
    print('cluster sizes in ascending order clust_size:clust_index:')
    for i in range(0,nC):
        print('{}:{}'.format(mCs[i],i),' ',end='')
    print('\n')
else:
    print('cluster sizes in kmeans order clust_size:clust_index:')
    for i in range(0,nC):
        print('{}:{}'.format(mC[i],i),' ',end='')
    print('\n')

ffnames = [] # to store full path (which exceeds fnames array elem size)

f = open('ffnames.txt','w')
pflag = 0
if args.inter == 'i':
    if int(args.perc)  == 0 and args.perc != '-0':
        perc = float(input('Select clusters < = percentile (usually 20-80):'))
    elif int(args.perc) == -1 or args.perc == '-0':
        perc = float(input('Select a specific cluster, enter -cluster_index:'))
        if perc > 0:
            print('non-negative value entered..fixing your error!')
            perc = -perc 
        if perc == 0: # users want cluster 0
            pflag = 1 
    else:
        print('bad option')
        sys.exit(1)
else:
    perc = float(args.perc)

if perc >= 0 and pflag == 0 and args.perc != '-0':
    print('percentile:',perc,'clustersize <=',np.percentile(mC,perc))
else:
    print('cluster:',int(abs(perc)),'clustersize:',mC[int(abs(perc))])

if perc >= 0 and pflag == 0 and args.perc != '-0': # process on normal percent
    for i in range(0,nC):
        if mC[i] <= np.percentile(mC,perc): # select clusters <=  perc 
            for j in membersC[i]:
                ffnames.append(os.path.join(prefixP,fnames[j]))
else:        # process nominated cluster 
    perc = int(abs(perc)) # we don't need the -ve sign any more
    for j in membersC[perc]:
        ffnames.append(os.path.join(prefixP,fnames[j]))
ffnames.sort()

for i in ffnames:
    f.write(i+'\n')
print(len(ffnames),'cluster imgfiles saved as ffnames.txt')

f.close()

# research section
whr = []
for i in range(len(fnames)):
   whr.append(whereinC(i,membersC,fnames))
if imgfull == 1: imgdist = 7 # add imgfull options to end of imgdist vals
if imgfull == 1 and img_bw == 1: imgdist = 8
with open('fn_'+str(imgdist)+'.pkl','wb') as fpkl:
   fpkl.write(pickle.dumps(whr))
