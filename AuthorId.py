'''
Created on Nov 1, 2015

@author: Mohamed Zahran
'''
from onlineLearning_HW2 import *
from Helper import Helper

def tryOnlineLearn():
    testShare = 0.25
    valShare = 0.4
    trainingFeatures, trainingLabels = Helper.parseGenderBlogDataset('blog-gender-dataset.csv')   
    if(FEATURE_TYPE == 'word2vec'):
        trainingFeatures = readDataVecs('genderBlogDatasetVectors.txt')    
    else:
        vocab = buildVocab(trainingFeatures)
        trainingFeatures = data2bow(trainingFeatures,vocab) 
                   
    tln = len(trainingFeatures)
    valData = trainingFeatures[0:int(tln*valShare)]
    valLabel = trainingLabels[0:int(tln*valShare)]
    vln = len(valData)
    testData = trainingFeatures[vln:int(vln+tln*testShare)]
    testLabel = trainingLabels[vln:int(vln+tln*testShare)]
    ln = len(valData)+len(testData)
    trainingFeatures = trainingFeatures[ln:]
    trainingLabels = trainingLabels[ln:]        
    
    
    model = avgPerceptron(trainingFeatures, trainingLabels)
    
    predictionsTraining = classify(trainingFeatures, trainingLabels, model, trainingFeatures, trainingLabels)
    predictionsVal = classify(valData, valLabel, model, trainingFeatures, trainingLabels)
    predictionsTest = classify(testData, testLabel, model, trainingFeatures, trainingLabels)
    
    print('Training set:')
    res = checkPerformance(trainingLabels, predictionsTraining) 
    print(res)   
    print('Validation set:')
    res = checkPerformance(valLabel, predictionsVal)
    print(res)    
    print('Test set:')
    res = checkPerformance(testLabel, predictionsTest)
    print(res)
  
def paramSelectOnlineLearning():
    
    writer = open('results_log.txt','w')
    # reading the training data, validation data, test data
    print ('>> Parsing all data sets ...')
    testShare = 0.25
    valShare = 0.25    
    trainingFeatures, trainingLabels = Helper.parseGenderBlogDataset('blog-gender-dataset.csv')
       
    print ('>> Building vocabulary ...')
    #vocab = buildVocab(trainingData, valData)        
         
    for feat in FEATURE_TYPE_LIST:
        global FEATURE_TYPE
        FEATURE_TYPE = feat
        
        if(feat == 'word2vec'):
            trainingFeatures = readDataVecs('genderBlogDatasetVectors.txt')    
                         
        for boolean in BOOLEAN_TYPE_LIST:
            global BOOLEAN_TYPE
            BOOLEAN_TYPE = boolean
             
            if(feat != 'word2vec'):
                vocab = buildVocab(trainingFeatures)
                trainingFeatures = data2bow(trainingFeatures,vocab) 
                
            tln = len(trainingFeatures)
            valData = trainingFeatures[0:int(tln*valShare)]
            valLabel = trainingLabels[0:int(tln*valShare)]
            vln = len(valData)
            testData = trainingFeatures[vln:int(vln+tln*testShare)]
            testLabel = trainingLabels[vln:int(vln+tln*testShare)]
            ln = len(valData)+len(testData)
            trainingFeatures = trainingFeatures[ln:]
            trainingLabels = trainingLabels[ln:]                
             
            for typee in LEARNING_TYPE_LIST:
                for margin in MARGIN_LIST:
                    for maxIter in MAX_ITERATION_LIST:
                        for lrate in LEARNING_RATE_LIST:
                             
                            try:
                                global MARGIN
                                MARGIN = margin                                                    
                                  
                                global MAX_ITERATION
                                MAX_ITERATION = maxIter
                                  
                                global LEARNING_RATE
                                LEARNING_RATE = lrate
                                  
                                global LEARNING_TYPE
                                LEARNING_TYPE = typee
                                  
                                myStr = 'FEATURE_TYPE='+str(feat) +' BOOLEAN_FEATURES='+str(boolean) +' LEARNING_TYPE='+str(typee)+ ' MARGIN='+str(margin) + ' MAX_ITERATION='+str(maxIter)+' LEARNING_RATE='+str(lrate)
                                writer.write('\n'+myStr)    
                                writer.flush()  
                                print (myStr)
                                #print ('>> Starting training ...')
                                if(LEARNING_TYPE == 'p'):
                                    model = perceptron(trainingFeatures, trainingLabels)
                                elif(LEARNING_TYPE == 'avgP'):
                                    model = avgPerceptron(trainingFeatures, trainingLabels)
                                elif(LEARNING_TYPE == 'w'):
                                    model = winnow(trainingFeatures, trainingLabels)
                                else:
                                    model = kernelPerceptron(trainingFeatures, trainingLabels)
                                  
                                  
                                #print ('>> Making predictions ...')
                                predictionsTraining = classify(trainingFeatures, trainingLabels, model, trainingFeatures, trainingLabels)    
                                predictionsVal = classify(valData, valLabel, model, trainingFeatures, trainingLabels)
                                predictionsTest = classify(testData, testLabel, model, trainingFeatures, trainingLabels)
                                  
                                #print ('>> Calculating performance ...')
                                #print('Training set:')
                                res = checkPerformance(trainingLabels, predictionsTraining)
                                writer.write('\nTRAIN: '+res)
                                #print('Validation set:')
                                res = checkPerformance(valLabel, predictionsVal)
                                writer.write('\nVAL  : '+res)
                                #print('Test set:')
                                res = checkPerformance(testLabel, predictionsTest)
                                writer.write('\nTEST : '+res)
                                  
                                writer.flush()
                            except:
                                writer.write('>> Expection !')
                                writer.flush()
                                
                  
    print('DONE !') 
    
def main():
    
    #trainingData,trainingLabels,valData,valLables,testData,testLabels = Helper.parseData('train.csv', 'validation.csv', 'test.csv')
    #cVect = CountVectorizer()
    #xTrain = cVect.fit_transform(trainingData)    
    #tryOnlineLearn()
    paramSelectOnlineLearning()
    
    
main()