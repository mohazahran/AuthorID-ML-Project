'''
Created on Nov 2, 2015

@author: zahran
'''
import re

class Helper:
    
    ALPHABET_ONLY = True
    FILTER_STOPWORDS = True
    LOWER_CASING = True    
    USE_UTF8 = True
    
    @staticmethod
    def clean(str):
        #s = ' '.join(e.lower() for e in str if e.isalnum())
        #s = str.lower().replace(',','').replace('.','').replace('-',' ').replace('"','').replace('(','').replace(')','').replace(':','')
        #s=' '.join(s.split())
        str = str.lower().strip()
        str = re.sub(r'\d+\.\d+', ' NUM ', str) #5555.5555
        str = re.sub(r'\d+', ' NUM ', str) #55555        
        str = re.sub(r'[^a-zA-Z\']',' ',str)
        str = re.sub(r'\s+',' ',str)
        return str
    
    @staticmethod
    def parseMovieReviewsData(trainName, valName, testName): # read and parse datasets into lists
        trainingData = []
        trainingLabels = []
        valData = []
        valLables = []
        testData = []   
        testLabels = []
        
        tfile = open(trainName)
        for line in tfile:
            if(Helper.USE_UTF8): 
                try:                                        
                    line = unicode(line, "utf-8")
                except:
                    continue          
            parts = line.rsplit(',', 1)
            if(parts[1].strip() == '+'):
                trainingLabels.append(1)
            else:
                trainingLabels.append(-1)
            trainingData.append(Helper.clean(parts[0]))            
            
        vfile = open(valName)
        for line in vfile:
            if(Helper.USE_UTF8):
                try:                                        
                    line = unicode(line, "utf-8")
                except:
                    continue 
            parts = line.rsplit(',', 1)
            if(parts[1].strip() == '+'):
                valLables.append(1)
            else:
                valLables.append(-1)
            valData.append(Helper.clean(parts[0]))         
            
        tfile = open(testName)
        for line in tfile:
            if(Helper.USE_UTF8):
                try:                                        
                    line = unicode(line, "utf-8")
                except:
                    continue 
            parts = line.rsplit(',', 1)
            if(parts[1].strip() == '+'):
                testLabels.append(1)
            else:
                testLabels.append(-1)
            testData.append(Helper.clean(parts[0]))         
            
        return trainingData,trainingLabels,valData,valLables,testData,testLabels    
    
    
    @staticmethod
    def parseGenderBlogDataset(fileName):
        trainingData = []
        trainingLabels = []
        data = open(fileName,'r')
        example = ''
        for line in data:            
            line = line.strip()            
            if(line != ''):
                example += line
            else:
                parts = example.rsplit(',', 1)
                label = parts[-1].strip()
                rawData = []
                cleanData = Helper.clean(parts[0])
                example = ''                
                if(label == 'M'):
                    trainingLabels.append(1)
                    trainingData.append(cleanData)
                elif(label == 'F'):
                    trainingLabels.append(-1)
                    trainingData.append(cleanData)
                else:                    
                    continue
#         w = open('genderBlogDataset.txt','w')
#         for line in trainingData:
#             w.write(line+'\n')
#         w.close()
        return trainingData, trainingLabels
                
            
            
    
    
def main():
    Helper.parseGenderBlogDataset('blog-gender-dataset.csv')  
    
    
if __name__ == '__main__':
    main()
    
    
    
    