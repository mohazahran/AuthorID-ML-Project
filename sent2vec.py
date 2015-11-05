'''
Created on Nov 4, 2015

@author: Mohamed Zahran
'''
from test.test_pep277 import filenames
'''
Created on Oct 5, 2015

@author: Mohamed Zahran
'''
import numpy as np

def clean(str):
    #s = ' '.join(e.lower() for e in str if e.isalnum())
    s = str.lower().replace(',','').replace('.','').replace('-',' ').replace('"','').replace('(','').replace(')','').replace(':','')
    s=' '.join(s.split())
    return s

def parseAllData(trainName, valName, testName): # read and parse datasets into lists
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

def parseData(fileName):    
    data = []
    read = open(fileName,'r')
    for line in read:
        data.append(line.strip())
    return data

def parseWordVec (VECTOR_PATH, WORD2VEC, MAX):
    print 'Start loading vectors ...'    
    vecDic = {}    
    vectors = None
    fin = open(VECTOR_PATH, "rb")  
    header = fin.readline()
    vocab_size, vector_size = map(int, header.split())    
    if(WORD2VEC): #for CBOW or Skip-gram
        binary_len = np.dtype(np.float32).itemsize * vector_size
    else: # for GloVe
        binary_len = np.dtype(np.float64).itemsize * vector_size
    threshold = min(vocab_size,MAX)
    for line_no in xrange(threshold):                
        word = ''
        while True:
            ch = fin.read(1)
            if ch == ' ' and WORD2VEC==1:
                break
            elif ch == '#' and WORD2VEC==0:
                break
            word += ch    
        if(WORD2VEC):       
            vector = np.fromstring(fin.read(binary_len), np.float32)
        else:                                                       
            vector = np.fromstring(fin.read(binary_len), np.float64)                   
        word = word.strip()                          
        vecDic[word] = vector        
    print 'finished loading vectors ...'
    return vecDic




def writeSentVecToFile(vecDic, data, fileName):
    writer = open(fileName,'w')  
    for line in data:       
        words = line.split(' ')
        sentVec = np.array([0]*300)
        minVec = [100000]*300
        maxVec = [-100000]*300
        finalVec = []
        cnt = 1        
        for w in words:
            if(w in vecDic):
                minVec = [(vecDic[w][i]) if(vecDic[w][i]<minVec[i]) else minVec[i] for i in range(len(vecDic[w]))]
                maxVec = [(vecDic[w][i]) if(vecDic[w][i]>maxVec[i]) else maxVec[i] for i in range(len(vecDic[w]))]
                sentVec = sentVec + vecDic[w]
                cnt += 1
        sentVec = sentVec/float(cnt)
        finalVec.append(minVec)
        finalVec.append(sentVec.tolist())
        finalVec.append(maxVec)
        #sentVec = np.append(np.array(minVec), sentVec/float(cnt),np.array(maxVec))
        writer.write(str(finalVec).strip().replace('[','').replace(']','')+'\n')
        #writer.write(str(finalVec))
        #writer.flush()                
    writer.close()




def main():
    VECTOR_PATH = 'D:/Career/Work/RDI/EnglishWordEmbeddings/GoogleNews-vectors-negative300.bin'
    vecDic = parseWordVec (VECTOR_PATH, 1, 1000000)
    data = parseData('genderBlogDataset.txt')
    writeSentVecToFile(vecDic, data, 'genderBlogDatasetVectors.txt')
    print 'DONE !'
#     TRAIN_FILE_NAME = "train.csv"
#     VALID_FILE_NAME = "validation.csv"
#     TEST_FILE_NAME = "test.csv"
#     trainingData,trainingLabels,valData,valLables,testData,testLabels = parseAllData(TRAIN_FILE_NAME,VALID_FILE_NAME,TEST_FILE_NAME)
#     vecDic = parseWordVec (VECTOR_PATH, 1, 4000000)
#     trainingVecs = 'trainingSent2vec.txt'
#     valVecs = 'valSent2vec.txt'
#     testVecs = 'testSent2vec.txt'
#     writeSentVecToFile(vecDic, trainingData, trainingVecs)
#     writeSentVecToFile(vecDic, valData, valVecs)
#     writeSentVecToFile(vecDic, testData, testVecs)









 
    

if __name__ == '__main__':
    main()