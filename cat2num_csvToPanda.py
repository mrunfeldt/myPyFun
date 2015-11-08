# Given location of csv file, opens file and assigns numeric values for catagories
# Save as dataFrame using pickle
# MJrunfeldt 2015_10_01

savePath = '/home/mel/kaggleLetter/processedData/'
loadPath = '/home/mel/kaggleLetter/train.csv'

import pandas as pd
import os
import numpy as np
import copy
import array

# # Reassign catagory variables to numbers. Also, 
# ID whether column is composed of string or int values
newFrame=pd.read_csv(loadPath) # load the whole file
nSamp = 10; catID=[];
for xcol in range(0,2):#len(newFrame.columns)):
 aCol=newFrame[newFrame.columns[xcol]] # full single column
 dumVar = []
 for a in range(0,nSamp): # Search initial entries from strings
  dumVar.append(int(isinstance(aCol[a],str))) # is first value a string? 0 = no
 if sum(dumVar)>0: # if strings exist
  vLog = np.array([[copy.deepcopy(aCol.values[0]),1]]) 
  aCol.values[0] = 1; catVals = 1; 
  for b in range(1,len(aCol)): # ID unique strings and assign value
   brak = 0; c=0
   while brak == 0 and c<catVals: #search for previous observation
     if aCol.values[b]== vLog[c,0]: #hit 
      aCol.values[b]=copy.deepcopy(vLog[c,1])
      brak = 1
     else: # new variable 
      c = copy.deepcopy(c)+1
   if brak == 0: # if no match was found, make new variable   
    catVals=copy.deepcopy(catVals)+1 # value to replace string
    vLog = np.vstack((copy.deepcopy(vLog),copy.deepcopy([aCol.values[b],catVals])))
    aCol.values[b]=copy.deepcopy(catVals)  
   # END (a) each row entry in "xcol" column
  # END IFF strings exist 
  newFrame[newFrame.columns[xcol]]=aCol # REPLACE PREVIOUS COLUMN
# # # END replacing cats with nums in single Column # #   
  
 if 'catLog' in locals() and 'vLog' in locals(): # if exist
  catLog =  np.vstack((copy.deepcopy(catLog),copy.deepcopy(vLog)))
  catID.append(1); del vLog
 elif 'catLog' not in locals() and 'vLog' in locals():
  catLog = copy.deepcopy(vLog);  # first instance
  catID.append(1); #del vLog
 else: # catLog exists, but xcol is numeric (no vLog)
   catID.append(0) # 
 print('column ' + str(xcol) +' out of ' + str(len(newFrame.columns)) ) #print update
    
newFrame.to_pickle(savePath+'newFrame.p') # save using pickle
#catLog.to_pickle(savePath+'catLog.p') # save using pickle
np.save(savePath+'catLog',catLog,delimiter=",") #save as numpy array 
#catID.to_pickle(savePath+'catID.p') # save using pickle
np.save(savePath+'catID',np.asarray(catID)) #save as numpy array 

         
