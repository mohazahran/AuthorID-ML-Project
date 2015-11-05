'''
Created on Oct 6, 2015

@author: Mohamed Zahran
'''
'''
Created on Sep 29, 2015

@author: Mohamed Zahran
'''
import numpy as np
import sys

MAX_ITERATION_LIST = [5,50,100]
# MAX_ITERATION = 1
LEARNING_RATE_LIST = [0.05,0.1,0.3]
# LEARNING_RATE = 0.01
LEARNING_TYPE_LIST = ['p','avgP','w']
# LEARNING_TYPE = 'kernelP'
FEATURE_TYPE_LIST = ['unigram','bigram', 'both', 'word2vec']
# FEATURE_TYPE = 'unigram'
BOOLEAN_TYPE_LIST = [0,1]
# BOOLEAN_TYPE = 1
MARGIN_LIST = [0,0.05,0.1,0.2]
# MARGIN = 0.1

MAX_ITERATION = 50
LEARNING_RATE = 0.1
LEARNING_TYPE = 'avgP'
FEATURE_TYPE = 'word2vec'
BOOLEAN_TYPE = 0
MARGIN = 0.1




def clean(str):
    #s = ' '.join(e.lower() for e in str if e.isalnum())
    s = str.lower().replace(',','').replace('.','').replace('-',' ').replace('"','').replace('(','').replace(')','').replace(':','')
    s=' '.join(s.split())
    return s

def parseData(trainName, valName, testName): # read and parse datasets into lists
    trainingData = []
    trainingLabels = []
    valData = []
    valLables = []
    testData = []   
    testLabels = []
    
    tfile = open(trainName)
    for line in tfile:
        parts = line.rsplit(',', 1)
        if(parts[1].strip() == '+'):
            trainingLabels.append(1)
        else:
            trainingLabels.append(-1)
        trainingData.append(clean(parts[0]))            
        
    vfile = open(valName)
    for line in vfile:
        parts = line.rsplit(',', 1)
        if(parts[1].strip() == '+'):
            valLables.append(1)
        else:
            valLables.append(-1)
        valData.append(clean(parts[0]))         
        
    tfile = open(testName)
    for line in tfile:
        parts = line.rsplit(',', 1)
        if(parts[1].strip() == '+'):
            testLabels.append(1)
        else:
            testLabels.append(-1)
        testData.append(clean(parts[0]))         
        
    return trainingData,trainingLabels,valData,valLables,testData,testLabels


def find_ngrams(lst, n):    
    myNgrams = [' '.join(lst[i:i+n]) for i in range(len(lst) - n)]
    return myNgrams
        

def buildVocab(trainingData):
    vocab={}
    freq={}
    id = 0
    if(FEATURE_TYPE == 'both'):        
        for item in trainingData:            
            tokens1 = find_ngrams(item.split(),1)
            tokens2 = find_ngrams(item.split(),2)
            for t in tokens1:
                if(t not in vocab):
                    vocab[t.strip()]=id
                    id += 1                                
            for t in tokens2:
                if(t not in vocab):
                    vocab[t.strip()]=id
                    id += 1
        

            
    else:
        window = 1
        if(FEATURE_TYPE == 'bigram'):
            window = 2                            
        for item in trainingData:
            #tokens = item.split()
            tokens = find_ngrams(item.split(),window)
            for t in tokens:
                if(t not in vocab):
                    vocab[t.strip()]=id
                    id += 1      
                                      
    return vocab
        
def data2bow(data, vocab):
    window = 1
    if(FEATURE_TYPE == 'bigram'):
        window = 2
        
    dataBow = []
    for item in data:
        bow = [0]*(len(vocab)+1)
        bow[-1]=1
        for t in find_ngrams(item.split(),window):        
            if(t.strip() in vocab):
                if(BOOLEAN_TYPE == 0):
                    bow[vocab[t]] += 1
                else:
                    bow[vocab[t]] = 1
        bow = np.array(bow)
        dataBow.append(bow)
    return dataBow
    

def perceptron(featureSet, labels):
    length = len(featureSet[0])    
    weights = np.array([0]*(length))
    for k in range(MAX_ITERATION):        
        for i in range(len(featureSet)):                        
            #prediction = np.sign(np.dot(weights,featureSet[i]))
            dotp = np.dot(weights,featureSet[i])
            if(dotp > MARGIN):
                prediction = 1
            else:
                prediction = -1
            if(dotp < (-1*MARGIN)):
                prediction = -1
            else:
                prediction = 1
            if(prediction != labels[i]):
                weights = weights + LEARNING_RATE*labels[i]*featureSet[i]
    return weights

def promotion (weights, features):
    for x in range(len(weights)):
        if(features[x]!=0):
            weights[x] *= (float(1.0)/float(LEARNING_RATE))
    return weights

def demotion (weights, features):
    for x in range(len(weights)):
        if(features[x]!=0):
            weights[x] /= (float(1.0)/float(LEARNING_RATE))
    return weights

def winnow(featureSet, labels):
    theta = len(featureSet[0])    
    weights = np.array([1]*(theta))
    for k in range(MAX_ITERATION):        
        for i in range(len(featureSet)):                        
            dot = np.dot(weights,featureSet[i])            
            if(dot >= theta):
                pred = 1
            else:
                pred = -1
            if(pred != labels[i]):            
                if(dot < theta and labels[i]==1):
                    weights = promotion(weights, featureSet[i]) 
                if(dot >= theta and labels[i]==-1):
                    weights = demotion(weights, featureSet[i])
    return weights

def avgPerceptron(featureSet, labels):
    length = len(featureSet[0])    
    weights = np.array([0]*(length))
    avgWeights = np.array([0]*(length))
    for k in range(MAX_ITERATION):        
        for i in range(len(featureSet)):                        
            #prediction = np.sign(np.dot(weights,featureSet[i]))
            dotp = np.dot(weights,featureSet[i])
            if(dotp > MARGIN):
                prediction = 1
            else:
                prediction = -1
            if(dotp < (-1*MARGIN)):
                prediction = -1
            else:
                prediction = 1
            if(prediction != labels[i]):
                weights = weights + LEARNING_RATE*labels[i]*featureSet[i]
            
            avgWeights = avgWeights + weights
                            
    return avgWeights


def kernel(v1, v2):    
    C = 0.1
    d = 2
    ker = (np.dot(v1,v2) + C)**2
    return ker
    

def kernelPerceptron(featureSet, labels):
    length = len(featureSet[0])    
    mistakes = {}        
        
    for k in range(MAX_ITERATION):  
        print('iter# ',k)      
        for i in range(len(featureSet)):     
            combined = 0
            for vecIdx, count in mistakes.iteritems():
                combined = combined + (LEARNING_RATE * count * labels[vecIdx] * kernel(featureSet[vecIdx],featureSet[i]))
                                        
            if(combined > MARGIN):
                prediction = 1
            else:
                prediction = -1
            if(combined < (-1*MARGIN)):
                prediction = -1
            else:
                prediction = 1
                
            if(prediction != labels[i]):                                
                if(i not in mistakes):
                    mistakes[i] = 1
                else:
                    mistakes[i] += 1
    return mistakes
    
            
    
    
def classify(data, labels, model, trainingBow, trainingLabels):
    predictions = []
    theta = len(model)
    for example in data:        
        if(LEARNING_TYPE=='w'):
            dot = np.dot(model,example)
            if(dot >= theta):
                predictions.append(1)
            else:
                predictions.append(-1)
                
                
                
        elif(LEARNING_TYPE =='kernelP'):
            combined = 0
            for vecIdx, count in model.iteritems():
                combined = combined + (LEARNING_RATE * count * trainingLabels[vecIdx] * kernel(trainingBow[vecIdx],example))
                prediction = np.sign(combined)
            predictions.append(prediction)
                
                
                
        else:
            dot = np.dot(model,example)
            prediction = np.sign(dot)
            predictions.append(prediction)
    return predictions
        

def checkPerformance(labels, predictions):
    if(len(labels) != len(predictions)):
        print 'ERROR: sizes dont match'
        return
    tp=0
    fp=0
    tn=0
    fn=0
    
    for i in range(len(labels)):
        if(labels[i] == predictions[i]):
            if(predictions[i] == 1):
                tp += 1
            else:
                tn += 1
        else:
            if(predictions[i] == 1):
                fp += 1
            else:
                fn += 1
    recall    = float(tp) / float (tp + fn) 
    precision = float(tp) / float (tp + fp)
    Fscore    = float(2*precision*recall) / float(precision+recall)        
    accuracy  = float(tp+tn) / float(tp+fp+tn+fn)
    
    myStr = "Recall="+str(recall)+" Precision="+str(precision)+" Fscore="+str(Fscore)+ " Accuracy="+str(accuracy)
    return myStr 
        
def readDataVecs(fileName):
    train = []
    data = open(fileName,'r')
    for line in data:
        nums = line.split(',')
        tmp = []
        for n in nums:
            tmp.append(float(n))        
        vec = np.array(tmp)        
        train.append(vec)
    return train
    
def sent2VectFeatures():
    train = []
    val = []
    test = []
    
    data = open('trainingSent2vec.txt','r')
    for line in data:
        nums = line.split(',')
        tmp = []
        for n in nums:
            tmp.append(float(n))        
        vec = np.array(tmp)        
        train.append(vec)
    
    data = open('valSent2vec.txt','r')
    for line in data:
        nums = line.split(',')        
        tmp = []
        for n in nums:
            tmp.append(float(n))        
        vec = np.array(tmp)  
        val.append(vec)
        
    data = open('testSent2vec.txt','r')
    for line in data:
        nums = line.split(',')
        tmp = []
        for n in nums:
            tmp.append(float(n))        
        vec = np.array(tmp)  
        test.append(vec)
        
    return train, val, test
        
def parseArgs(args):  
    args_map = {}
    curkey = None
    for i in xrange(1, len(args)):
        if args[i][0] == '-':
            args_map[args[i]] = True
            curkey = args[i]
        else:
            assert curkey
            args_map[curkey] = args[i]
            curkey = None
    return args_map

def validateInput(args):
    args_map = parseArgs(args)    
    if '-x' in args_map:
        max = int(args_map['-x'])
        global MAX_ITERATION
        MAX_ITERATION = max 
    
    if '-l' in args_map:
        learn = (args_map['-l'])
        global LEARNING_TYPE
        LEARNING_TYPE = learn
      
    if '-r' in args_map:
        rate = float(args_map['-r'])
        global LEARNING_RATE
        LEARNING_RATE = rate
    
    if '-f' in args_map:
        feat = (args_map['-f'])
        global FEATURE_TYPE
        FEATURE_TYPE = feat 
        
    if '-b' in args_map:
        b = int(args_map['-b'])
        global BOOLEAN_TYPE
        BOOLEAN_TYPE = b
        
    if '-m' in args_map:
        m = float(args_map['-m'])
        global MARGIN
        MARGIN = m
        
def main():
    TRAIN_FILE_NAME = "train.csv"
    VALID_FILE_NAME = "validation.csv"
    TEST_FILE_NAME = "test.csv"
    
    validateInput(sys.argv)
    myStr = 'FEATURE_TYPE='+str(FEATURE_TYPE) +' BOOLEAN_FEATURES='+str(BOOLEAN_TYPE) +' LEARNING_TYPE='+str(LEARNING_TYPE)+ ' MARGIN='+str(MARGIN) + ' MAX_ITERATION='+str(MAX_ITERATION)+' LEARNING_RATE='+str(LEARNING_RATE)
    print(myStr)
      
    print ('>> Parsing all data sets ...')
    trainingData,trainingLabels,valData,valLables,testData,testLabels = parseData(TRAIN_FILE_NAME,VALID_FILE_NAME,TEST_FILE_NAME)
    
    print ('>> Extracting features ...')
    if(FEATURE_TYPE == 'word2vec'):
        trainingBow, valBow, testBow  = sent2VectFeatures()
    else:
        vocab = buildVocab(trainingData)
        trainingBow = data2bow(trainingData,vocab)    
        valBow = data2bow(valData,vocab)
        testBow = data2bow(testData,vocab)   
    
    print ('>> Starting training ...')   
    if(LEARNING_TYPE == 'p'):
        model = perceptron(trainingBow, trainingLabels)
    elif(LEARNING_TYPE == 'avgP'):      
        model = avgPerceptron(trainingBow, trainingLabels)
    elif(LEARNING_TYPE == 'w'):    
        model = winnow(trainingBow, trainingLabels)
    else:  
        model = kernelPerceptron(trainingBow, trainingLabels)     
    
    print ('>> Making predictions ...')
    predictionsTraining = classify(trainingBow, trainingLabels, model, trainingBow, trainingLabels)    
    predictionsVal = classify(valBow, valLables, model, trainingBow, trainingLabels)
    predictionsTest = classify(testBow, testLabels, model, trainingBow, trainingLabels)
      
    print ('>> Calculating performance ...')
    print('Training set:')
    res = checkPerformance(trainingLabels, predictionsTraining) 
    print(res)   
    print('Validation set:')
    res = checkPerformance(valLables, predictionsVal)
    print(res)    
    print('Test set:')
    res = checkPerformance(testLabels, predictionsTest)
    print(res)
    

if __name__ == '__main__':
    main()