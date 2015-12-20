
# # # # # # FEED FORWARD NEURAL NET FOR CLASSIFICATION # # # # # #
#(1) Load Olivetti face dataset and uses PCA to reduce dimensionality
#(2) Divide data into Train and Test sets, and encodes output using "LabelBinarizer"
#(3) Construct Feed Froward NN using Pybrain, Supervised DataSet
#    * Supervised DataSet with encoded outputs used in leiu of Classification DataSet
#     so that new (test) data with UNKNOWN targets can be predicted
#(4) Report accuracy as percent correct


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

# # # # LOAD DATA # # # # 
oFace = datasets.fetch_olivetti_faces() # load data
X,y = oFace.data, oFace.target # extract data: [face x 1D vectorized image]
nFaces = len(np.unique(y)); nObs = len(y) # number of unique faces, total observations

# # # # SET PARAMETERS # # # #
perTrain = 0.85 # fraction of data to use for training
perTest = 1-perTrain
dimRed = 0.99  # percent variance retained for PCA dimensionality reduction

#=====================DATA PREPROCESSING=======================================
# # # Perform PCA to reduce dimensionality of data (in space) # # # 
pcInit = PCA(); pcaOne=pcInit.fit(X)
eigNorm = pcaOne.explained_variance_ratio_;  # percent variance (normalized eigenvalues)
nComps = [ n for n,i in enumerate(np.cumsum(eigNorm)) if i<dimRed ][-1:]
nComps = int(nComps[0])
print( "Reduced Dims from %d to %d" % (X.shape[1],nComps) )
pcaTwo = PCA(n_components=nComps); XlowD = pcaTwo.fit_transform(X)
del X; X = XlowD; del XlowD

# # # Randomly divide data into training and test portions # # # # # 
trainIdx = random.sample(range(nObs),round(nObs*perTrain) ) #Test Indicies
testIdx = np.setdiff1d(np.array(range(nObs)),trainIdx, assume_unique=True) # not Test indicies
trainIn = X[trainIdx,:]; trainY = y[trainIdx] # training
testIn = X[testIdx,:]; testY = y[testIdx] # test data (use to obtain accuracy)

# # # ENCODE TARGET CLASS AS BINARY FOR CLASSIFICATION # # #
lb = preprocessing.LabelBinarizer()
lb.fit(y)
trainOut = lb.transform(trainY)
testOut = lb.transform(testY)
#==============================================================================

#===============================FF ANN=========================================
# # # # SET NETWORK PARAMETERS # # # #
nIn = X.shape[1] ; # number of input variables
nOut = nFaces; # number of output variables: one per class
nHid = round(nIn+nOut/2) ; # number of hidden layers: should be optimized

# # # # BUILD TRAINING DATASET # # # # #
dsTrain = SupervisedDataSet(nIn,nOut)
for i in range(len(trainIdx)):
    dsTrain.addSample(trainIn[i,:],trainOut[i,:])

# # # BUILD FEED FORWARD NETWORK # # # # #
fnn = buildNetwork( nIn,nHid,nFaces, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dsTrain, momentum=0.1, verbose=True, weightdecay=0.01)

# # # # TRAIN NETWORK # # # # 
trainer.trainUntilConvergence(dataset=dsTrain,
                              maxEpochs=100,
                              verbose=False,
                              continueEpochs=10,
                              validationProportion=0.2) 
                              
# # # # PREDICT TEST DATA AND REPORT PERCENT CORRECT # # #                 
hit = 0; N = testIn.shape[0] ;
pred = np.empty(testIn.shape[0])
for i in range(testIn.shape[0]):                              
    pred[i]=fnn.activate(testIn[i,:]).argmax(axis=0)                           
    if pred[i] == testY[i]:
        hit += 1

perCorr = hit/N*100                                                                          
print("Perecent Correct = %d " %perCorr)       
        
        
#==============================================================================        
# # Optional: Reproject Vector into 2D to check(visualize) faces # # 
def twoToOne(data,d): # only works for square matrices
    twoD = np.empty((d,d))
    for a in range(0,d):
        twoD[a,:] = data[d*a:d*(a+1)]
    return twoD
#dum = twoToOne(X[23,:],64);plt.imshow(dum) # plot a 2D face        
