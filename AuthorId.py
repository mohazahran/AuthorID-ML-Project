'''
Created on Nov 1, 2015

@author: Mohamed Zahran
'''
import onlineLearning_HW2
from Helper import Helper
import random

def tryOnlineLearn():
    testShare = 0.25
    valShare = 0.25
    trainingFeatures, trainingLabels = Helper.parseGenderBlogDatasetWithLabels('blog-gender-dataset.csv')  
    #trainingFeatures = random.shuffle(trainingFeatures)  
    if(onlineLearning_HW2.FEATURE_TYPE == 'word2vec'):
        trainingFeatures = onlineLearning_HW2.readDataVecs('genderBlogDatasetVectors.txt')    
    else:
        vocab = onlineLearning_HW2.buildVocab(trainingFeatures)
        trainingFeatures = onlineLearning_HW2.data2bow(trainingFeatures,vocab)
           
                   
    tln = len(trainingFeatures)
    valData = trainingFeatures[0:int(tln*valShare)]
    valLabel = trainingLabels[0:int(tln*valShare)]
    vln = len(valData)
    testData = trainingFeatures[vln:int(vln+tln*testShare)]
    testLabel = trainingLabels[vln:int(vln+tln*testShare)]
    ln = len(valData)+len(testData)
    trainingFeatures = trainingFeatures[ln:]
    trainingLabels = trainingLabels[ln:]        
    
    
    model = onlineLearning_HW2.avgPerceptron(trainingFeatures, trainingLabels)
    
    predictionsTraining = onlineLearning_HW2.classify(trainingFeatures, trainingLabels, model, trainingFeatures, trainingLabels)
    predictionsVal = onlineLearning_HW2.classify(valData, valLabel, model, trainingFeatures, trainingLabels)
    predictionsTest = onlineLearning_HW2.classify(testData, testLabel, model, trainingFeatures, trainingLabels)
    
    print('Training set:')
    res = onlineLearning_HW2.checkPerformance(trainingLabels, predictionsTraining) 
    print(res)   
    print('Validation set:')
    res = onlineLearning_HW2.checkPerformance(valLabel, predictionsVal)
    print(res)    
    print('Test set:')
    res = onlineLearning_HW2.checkPerformance(testLabel, predictionsTest)
    print(res)
  
def paramSelectOnlineLearning():
    
    writer = open('results_log.txt','w')
    # reading the training data, validation data, test data
    print ('>> Parsing all data sets ...')
    testShare = 0.25
    valShare = 0.25    
    trainingData, trainingLabels = Helper.parseGenderBlogDatasetWithLabels('blog-gender-dataset.csv')  
       
    print ('>> Building vocabulary ...')
    #vocab = buildVocab(trainingData, valData)        
         
    bestVal = 0
    bestValParam = ''    
    
    bestTest = 0
    bestTestParam = ''    
    
    bestTrain = 0
    bestTrainParam = ''    
    
    for feat in onlineLearning_HW2.FEATURE_TYPE_LIST:
        global FEATURE_TYPE
        FEATURE_TYPE = feat               
                         
        for boolean in onlineLearning_HW2.BOOLEAN_TYPE_LIST:
            global BOOLEAN_TYPE
            BOOLEAN_TYPE = boolean
             
            if(feat != 'word2vec'):
                vocab = onlineLearning_HW2.buildVocab(trainingData)
                trainingFeatures = onlineLearning_HW2.data2bow(trainingData,vocab) 
                
            if(feat == 'word2vec'):
                trainingFeatures = onlineLearning_HW2.readDataVecs('genderBlogDatasetVectors.txt')    
                
            tln = len(trainingFeatures)
            valFeatures = trainingFeatures[0:int(tln*valShare)]
            valLabel = trainingLabels[0:int(tln*valShare)]
            vln = len(valFeatures)
            testFeatures = trainingFeatures[vln:int(vln+tln*testShare)]
            testLabel = trainingLabels[vln:int(vln+tln*testShare)]
            ln = len(valFeatures)+len(testFeatures)
            trainingFeatures = trainingFeatures[ln:]
            trainingOnlyLabels = trainingLabels[ln:]                
            
            #print ('>> Starting Training ...')
            for typee in onlineLearning_HW2.LEARNING_TYPE_LIST:
                for margin in onlineLearning_HW2.MARGIN_LIST:
                    for maxIter in onlineLearning_HW2.MAX_ITERATION_LIST:
                        for lrate in onlineLearning_HW2.LEARNING_RATE_LIST:
                             
                            #try:
                                #global onlineLearning_HW2.MARGIN
                                onlineLearning_HW2.MARGIN = margin                                                    
                                  
                                #global onlineLearning_HW2.MAX_ITERATION
                                onlineLearning_HW2.MAX_ITERATION = maxIter
                                  
                                #global onlineLearning_HW2.LEARNING_RATE
                                onlineLearning_HW2.LEARNING_RATE = lrate
                                  
                                #global onlineLearning_HW2.LEARNING_TYPE
                                onlineLearning_HW2.LEARNING_TYPE = typee
                                  
                                myStr = 'FEATURE_TYPE='+str(feat) +' BOOLEAN_FEATURES='+str(boolean) +' LEARNING_TYPE='+str(typee)+ ' MARGIN='+str(margin) + ' MAX_ITERATION='+str(maxIter)+' LEARNING_RATE='+str(lrate)
                                writer.write('\n'+myStr)    
                                writer.flush()  
                                print (myStr)
                                #print ('>> Starting training ...')
                                if(onlineLearning_HW2.LEARNING_TYPE == 'p'):
                                    model = onlineLearning_HW2.perceptron(trainingFeatures, trainingOnlyLabels)
                                elif(onlineLearning_HW2.LEARNING_TYPE == 'avgP'):
                                    model = onlineLearning_HW2.avgPerceptron(trainingFeatures, trainingOnlyLabels)
                                elif(onlineLearning_HW2.LEARNING_TYPE == 'w'):
                                    model = onlineLearning_HW2.winnow(trainingFeatures, trainingOnlyLabels)
                                else:
                                    model = onlineLearning_HW2.kernelPerceptron(trainingFeatures, trainingOnlyLabels)
                                  
                                  
                                #print ('>> Making predictions ...')
                                predictionsTraining = onlineLearning_HW2.classify(trainingFeatures, trainingOnlyLabels, model, trainingFeatures, trainingOnlyLabels)    
                                predictionsVal = onlineLearning_HW2.classify(valFeatures, valLabel, model, trainingFeatures, trainingOnlyLabels)
                                predictionsTest = onlineLearning_HW2.classify(testFeatures, testLabel, model, trainingFeatures, trainingOnlyLabels)
                                  
                                #print ('>> Calculating performance ...')
                                #print('Training set:')
                                res, trainAcc = onlineLearning_HW2.checkPerformance(trainingOnlyLabels, predictionsTraining)
                                writer.write('\nTRAIN: '+res)
                                #print('Validation set:')
                                res, valAcc = onlineLearning_HW2.checkPerformance(valLabel, predictionsVal)
                                writer.write('\nVAL  : '+res)
                                #print('Test set:')
                                res, testAcc = onlineLearning_HW2.checkPerformance(testLabel, predictionsTest)
                                writer.write('\nTEST : '+res)
                                  
                                writer.flush()
                                                                
                           # except:
                            #    writer.write('>> Expection !')
                            #    writer.flush()
                            
                                if(trainAcc > bestTrain):
                                    bestTrain = trainAcc
                                    bestTrainParam = myStr
                                if(valAcc > bestVal):
                                    bestVal = valAcc
                                    bestValParam = myStr
                                if(testAcc > bestTest):
                                    bestTest = testAcc
                                    bestTestParam = myStr
                                
    
    writer.write('BEST train param:\n'+bestTrainParam)            
    writer.write('BEST val param:\n'+bestValParam)
    writer.write('BEST test param:\n'+bestTestParam)
    print('DONE !') 
    
def main():
    
    #trainingData,trainingLabels,valData,valLables,testData,testLabels = Helper.parseData('train.csv', 'validation.csv', 'test.csv')
    #cVect = CountVectorizer()
    #xTrain = cVect.fit_transform(trainingData)    
    #tryOnlineLearn()
    paramSelectOnlineLearning()
    
    
main()