import numpy as np
import matplotlib.pyplot as plt
import csv
#import Main
import os
import Evaluation
import random



from torch.utils.data import Dataset as TorchDataset

def getProjectPath():
	return './'

def splitBySignals(dataStep):
    segments= []
    for input, target in dataStep:
        targetInt = np.argmax(Evaluation.addNoGestureSignal(target), 1)
        inds= np.where(targetInt[:-1]!= targetInt[1:])[0]
        lastInd = -1
        for ind in inds:
            if targetInt[ind] != np.max(targetInt):
                iSegment = input[lastInd+1:ind+1]
                tSegement = target[lastInd+1:ind+1]
                tSegement[0,:]=0
                tSegement[-1,:]=0
                segments.append((iSegment,tSegement))
                lastInd = ind
        ind = len(targetInt)-1
        iSegment = input[lastInd+1:ind+1]
        tSegement = target[lastInd+1:ind+1]
        tSegement[0,:]=0
        tSegement[-1,:]=0
        segments.append((iSegment,tSegement))
    return segments

def shuffleDataStep(dataStep, nFolds, nRepeat=1):
    segs = splitBySignals(dataStep)
    segs = segs * nRepeat
    random.shuffle(segs)
    segs = [ segs[i::nFolds] for i in range(nFolds) ]
    dataStep=[]
    for segList in segs:
        ind = np.concatenate([x[0] for x in segList],0)
        t   = np.concatenate([x[1] for x in segList],0)
        dataStep.append((ind,t))
    return dataStep

class UniHHIMUGestures(TorchDataset):
    

    def __init__(self, 
                 dataDir,
                 inputFiles, testFiles, train=True, 
                 inputGestures=[0,1,2,3,4,5,6,7,8,9], 
                 usedGestures=[0,1,2,3,4,5,6,7,8,9], 
                 useNormalized=2, 
                 shuffle=True, nFolds=4, nRepeat=1, noiseFactor=1, learnTreshold=False):
        """
        Args:
            inputFiles (list<string>): Files/people to be used in trainset.
            testFiles (list<string>): Files/people to be used in testset.
            train (bool): wether or not the get method will return train or test samples
            useNormalized (int): normalisation of input signal. 
                0: no normalisation
                1: scale each sensors std to 1
                2: scale each sensors max value to 1 
        """
        
        
        self.train = train
        self.useNormalized = useNormalized
        
        
        # ===================================================================
        # Create trainset
        # ===================================================================
        
        
        dataStep = []
        for fileName in inputFiles:
            ind, t  = createData(fileName, inputGestures,usedGestures, dataDir=dataDir)
            dataStep.append((ind,t))

        # calculate normalizers from train files
        inputs = np.concatenate([inputs for inputs, targets in dataStep])
        self.normalizer = np.ones(9)
        if self.useNormalized == 1:
            self.normalizer[0:3] = np.std(np.linalg.norm(inputs[:,0:3], None, 1))
            self.normalizer[3:6] = np.std(np.linalg.norm(inputs[:,3:6], None, 1))
            self.normalizer[6:9] = np.std(np.linalg.norm(inputs[:,6:9], None, 1))
        if self.useNormalized == 2:
            self.normalizer[0:3] = np.max(np.linalg.norm(inputs[:,0:3], None, 1))
            self.normalizer[3:6] = np.max(np.linalg.norm(inputs[:,3:6], None, 1))
            self.normalizer[6:9] = np.max(np.linalg.norm(inputs[:,6:9], None, 1))

            
        # if desired shuffle and rearrage the data in nFolds
        # each fold can contain gestures from each person in the trainset, but not from the testset person 
        if(shuffle):
            dataStep = shuffleDataStep(dataStep, nFolds=nFolds, nRepeat=nRepeat)


        self.train_data = []
        for ind, t in dataStep:

            ind[:,0:3] += np.random.normal(0,0.05 *noiseFactor, size=(len(ind),3))
            ind[:,3:6] += np.random.normal(0,0.5 * noiseFactor, size=(len(ind),3))
            ind[:,6:9] += np.random.normal(0,1.25 * noiseFactor, size=(len(ind),3))

            # if treshold shall be learned, another target signal needs to be added. 
            # No gestures signal is 1 if all other targets are 0 and -1 otherwise.
            if learnTreshold:
                t = np.append(t,1-2*t.max(1, keepdims=True),1)

            self.train_data.append((ind,t))
            
            
        # ===================================================================
        # Create testset
        # ===================================================================
        dataStep = []
        for fileName in testFiles:
            ind, t  = createData(fileName, inputGestures,usedGestures, dataDir=dataDir)
            dataStep.append((ind,t))
        
        if shuffle:
            test_data = shuffleDataStep(dataStep, nFolds=1, nRepeat=1)
        else:
            print("WARNING: set shuffle to true, there's a bug somewhere in the input scaling otherwise")
        self.test_data = test_data


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            inputs, targets = self.train_data[idx]
        else:
            inputs, targets = self.test_data[idx]
        
        inputs /= self.normalizer
        
        return inputs, targets





class DataSet(object):
    
    #fused = np.empty((0,0))
    #gyro = np.empty((0,0))
    #acc  = np.empty((0,0))
    #targets = np.empty((0,0))
    #means = np.empty((0,0))
    #stds = np.empty((0,0))
    #gestures = np.empty((0,0))
    
    
    def __init__(self, fused, gyro, acc, targets,means, stds, gestures):
        self.fused = fused
        self.gyro = gyro
        self.acc = acc 
        self.targets = targets
        self.means = means
        self.stds = stds
        self.gestures = gestures
        



       
    def plot(self, targetNr=2,normalized = -1):
        fig = plt.figure(figsize=(10,10))
        plt.clf()
        plt.subplot(411)
        plt.title('Fused')
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            plt.plot(self.fused[:,i],label=labels[i])
        #plt.plot(self.targets[:,targetNr],label='Target')
        plt.ylim(-1.5,1.5)
        plt.legend()
        
        plt.subplot(412)
        plt.title('Gyro')
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            plt.plot(self.gyro[:,i],label=labels[i])
        #plt.plot(self.targets[:,targetNr],label='Target')
        plt.legend()
        
        plt.subplot(413)
        plt.title('Acc')
        labels = ['X', 'Y', 'Z']
        for i in range(3):
            plt.plot(self.acc[:,i],label=labels[i])
        #plt.plot(self.targets[:,targetNr],label='Target')
        plt.legend()
        
        plt.subplot(414)
        plt.title('Marker and Target')
        labels = ['Marker', 'Target']
        plt.plot(self.targets[:,0], label=labels[0])
        plt.plot(self.targets[:,2], label=labels[1])
        plt.ylim(-0.5,1.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return fig
        
    def getData(self):
        return  np.concatenate((self.fused,self.gyro,self.acc,self.targets),1)
    
    def getFused(self):
        return self.fused
    
    def getAcc(self):
        return self.acc
    
    def getGyro(self):
        return self.gyro 
    
    def getDataForTraining(self, classNrs, targetNr=2, multiplier = 1, normalized = False, power = False):
        inputData = np.empty((0,0))
        stds = np.empty((0,0))
        inputData = self.fused
        stds = self.stds[0:3]
        inputData = np.append(inputData, self.gyro, 1)
        stds = np.append(stds,self.stds[3:6],0)
        inputData = np.append(inputData, self.acc, 1)
        stds = np.append(stds,self.stds[6:9],0)
        
        i = 0
        target = np.copy(self.targets)
        while i < len(self.targets):
            tLen = 0
            while i < len(self.targets) and self.targets[i,2] == 1:
                tLen = tLen + 1
                i = i+1
            if tLen != 0:

                dropArea = np.min([tLen-5,tLen*(1/4)])
                #target[i-dropArea:i,2]=0
                target[int(i-tLen):int(i-tLen+dropArea),2]=0
            i = i+1   
        
        readOutTrainingData = np.zeros((len(inputData),len(classNrs)))
        i = 0
        for classNr in classNrs:
            readOutTrainingData[:,i] = target[:,targetNr].T * self.gestures[classNr]
            i = i+1
        if normalized:
            inputData = inputData/stds
        if power:
            inputData = np.append(inputData, np.atleast_2d(normPower(inputData)).T, 1)
            inputData = np.append(inputData, np.atleast_2d(normRot(inputData)).T, 1)
            inputData = np.append(inputData, np.atleast_2d(normFused(inputData)).T, 1)
            
        data = inputData
        target = readOutTrainingData
        for i in range(1,multiplier):
            data = np.append(data,inputData,0)
            target = np.append(target,readOutTrainingData,0)
        return (data,target)
    
    def getAllSignals(self, gesture = -1, targetNr = 2):
        signals = []
        target = self.targets[:,targetNr]
        if self.gestures[gesture] != 0 or gesture == -1:
            changesT = np.where(target[:-1] != target[1:])[0] + 1
            lastInd = 0
            for ind in changesT:
                if target[lastInd] == 1:
                    signals.append(np.concatenate((self.fused[lastInd:ind,:],self.gyro[lastInd:ind,:],self.acc[lastInd:ind,:],np.atleast_2d(target[lastInd:ind]).T),1))
                lastInd = ind
        return signals
    
    def getMinusPlusDataForTraining(self, classNr ,targetNr=2, multiplier = 1):
        inputData, target = self.getDataForTraining(classNr, targetNr, multiplier, True)
        low_values_indices = target == 0  # Where values are low
        target[low_values_indices] = -1   
        return (inputData,target)
        
        
    def unnormalize(self):
        self.fused = np.add(np.multiply(self.fused,self.stds[0:3]),self.means[0:3])
        self.gyro = np.add(np.multiply(self.gyro,self.stds[3:6]),self.means[3:6])
        self.acc = np.add(np.multiply(self.acc,self.stds[6:9]),self.means[6:9])
        
    def writeToFile(self, fileName):
        np.savez(getProjectPath()+'dataSets/'+fileName,  \
                    fused=self.fused,gyro=self.gyro,acc=self.acc,targets=self.targets,means=self.means,stds=self.stds,gestures=self.gestures)
        
        
    
def normPower(X):
    print(X.shape)
    return np.linalg.norm(X[:,6:9], None, 1) 
def normRot(X):
    return np.linalg.norm(X[:,3:6], None, 1) 
def normFused(X):
    return np.linalg.norm(X[:,0:3], None, 1) 

        
def createDataSetFromFile(fileName, dataDir='dataSets/'):
    
    data = np.load(os.path.join(getProjectPath(),dataDir,fileName))
    fused = data['fused']
    gyro = data['gyro']
    acc = data['acc']
    targets = data['targets']
    means = data['means']
    stds = data['stds']
    gestures = data['gestures']
    return DataSet(fused, gyro, acc, targets,means, stds, gestures)


def appendDS(dataSets, usedGestures):
    result = (dataSets[0].getDataForTraining(usedGestures,2)[0],\
              dataSets[0].getDataForTraining(usedGestures,2)[1])
    for i in range(1,len(dataSets)):
        result = (np.append(result[0], \
                  dataSets[i].getDataForTraining(usedGestures,2)[0],0), \
                  np.append(result[1], \
                  dataSets[i].getDataForTraining(usedGestures,2)[1],0))
    return result

def createData(dataSetName, inputGestures, usedGestures, scaleFactor = 1, dataDir='dataSets/'):
    dataSets= []
    for gesture in inputGestures:
        fullName = dataSetName + '_' +str(gesture) + '_' + 'fullSet.npz'
        dataSets.append(createDataSetFromFile(fullName, dataDir=dataDir))
    resultInputs,resultTargets = appendDS(dataSets, inputGestures)
    inds = np.where(np.in1d(inputGestures, usedGestures))[0]
    
    if scaleFactor != 1:
        scaledInputs = np.zeros((resultInputs.shape[0]*scaleFactor,resultInputs.shape[1]))
        scaledTargets= np.zeros((resultTargets.shape[0]*scaleFactor,resultTargets.shape[1]))
        for l,line in enumerate(resultInputs):
            for i in range(scaleFactor):
                scaledInputs[l*scaleFactor+i,:] = line
                scaledTargets[l*scaleFactor+i,:] = resultTargets[l,:]
        return (scaledInputs,scaledTargets[:,inds])
    else: 
        return (resultInputs,resultTargets[:,inds])
#def appendDataSets(ds1, ds2):
#    fused = np.append(ds1.fused, ds2.fused, 0)
#    gyro = np.append(ds1.gyro, ds2.gyro, 0)
#    acc =  np.append(ds1.acc, ds2.acc, 0)
#    targets = np.append(ds1.targets, ds2.targets, 0)
#    gestures = np.max(np.append(np.atleast_2d(ds1.gestures),np.atleast_2d(ds2.gestures),0),0)
#    return DataSet(fused, gyro, acc, targets,[], [], gestures)
    