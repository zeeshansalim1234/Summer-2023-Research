import logs
import numpy as np
import pandas as pd

label = 'label'
req1 = 'summary1'
req2 = 'summary2'
annStatus = 'AnnotationStatus'
fields = ['summary1','summary2','dependency','id1', 'id2','label']
from uncertaintySampling import leastConfidenceSampling,minMarginSampling,entropySampling

def predictLabels(cv,tfidf,clf,df_toBePredictedData):
    '''
    Passes the to be predicted dataset via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Predicts and returns the labels for the input data in a form of DataFrame.

    Parameters : 
    cv : Count Vectorizer Model
    tfidf : TFIDF Transformer Model
    clf : Trained Model 
    df_toBePredictedData (DataFrame) : To Be Predicted Data (Unlabelled Data)

    Returns : 
    df_toBePredictedData (DataFrame) : Updated To Be Predicted Data (Unlabelled Data), including prediction probabilities for different labels
    '''
    predictData = np.array(df_toBePredictedData.loc[:,[req1,req2]])
    
    predict_counts = cv.transform(predictData)
    predict_tfidf = tfidf.transform(predict_counts)
    predict_labels = clf.predict(predict_tfidf)
    predict_prob = clf.predict_proba(predict_tfidf)
    
    logs.writeLog ("\nTotal Labels Predicted : "+ str(len(predict_labels)))

    # contains predicted probs for all labels for each row 
    # so if there are 6 labels it will be like: [0.1, 0.14, 0.19, 0.14, 0.24, 0.19], where total adds to 1.0
    df_toBePredictedData['predictedProb'] = predict_prob.tolist() 
    # probablity of maxProbality label for that row
    df_toBePredictedData['maxProb'] = np.amax(predict_prob,axis=1)
    
    return df_toBePredictedData # Returns the unlaballed data with 2 new columns: `predictedProb` and `maxProb`

def analyzePredictions(args,df_predictions):
    '''
    Analyzis the predictions, samples the most uncertain data points and queries it from the oracle (original database/file) and updates dataframe accordingly.
    '''
    #df_manuallyAnnotated = pd.DataFrame(columns=['comboId','req1Id','req1',req1,'req2Id','req2',req2,label,annStatus])#Create an empty Dataframe to store the manually annotated Results

    """Intelligently Annotate""" 

    threshold = 0.80  # Replace with your desired threshold value

    # Filter rows based on the condition
    df_confident_predictions = df_predictions[df_predictions['maxProb'] > threshold]

    # Get the index values of the filtered rows
    confident_indexes = df_confident_predictions.index

    # Remove the filtered rows from df_predictions
    df_predictions.drop(index=confident_indexes, inplace=True)
    df_predictions.reset_index(drop=True, inplace=True)
    df_confident_predictions['annStatus'] = 'I'  # Mark all rows as intelligently annotated
    df_confident_predictions = df_confident_predictions[[req1,req2,label,annStatus]]

    """Annotate with Active Learning (Manual)""" 

    queryType = args.loc[0,'samplingType']
    df_userAnnot = pd.DataFrame(columns = fields)
    
    for field in [0,1,2,3,4,5]:
        iteration = 0
        logs.writeLog("\n\nIteration for field: "+str(field))
        #input("hit enter to proceed")

        # This selects `manualAnnotationsCount` (12) uncertain sample for each label, so total = manualAnnotationsCount * len(fields) = 72
        while iteration<int(args.loc[0,'manualAnnotationsCount']):  #while iteration is less than number of annotations that need to be done.
            if (len(df_predictions[df_predictions[label]==field ])>0):
                logs.writeLog("\n\nIteration : "+str(iteration+1))
                if queryType == 'leastConfidence':
                    indexValue = leastConfidenceSampling(df_predictions[df_predictions[label]==field ])
                elif queryType == 'minMargin':
                    indexValue = minMarginSampling(df_predictions[df_predictions[label]==field ])
                elif queryType == 'entropy':
                    indexValue =entropySampling(df_predictions[df_predictions[label]==field ])
            
                # indexValue is the id of the row with the most uncertain sample

                sample = df_predictions.loc[indexValue,:]  # gets the row with the most uncertain sample
                logs.writeLog("\n\nMost Uncertain Sample : \n"+str(sample))
                df_userAnnot = df_userAnnot.append({req1:sample[req1],req2:sample[req2],label:sample[label],annStatus:'M'},ignore_index=True)#df_userAnnot.append({'comboId':sample['comboId'],'req1Id':sample['req1Id'],'req1':sample['req1'],req1:sample[req1],'req2Id':sample['req2Id'],'req2':sample['req2'],req2:sample[req2],label:sample[label],annStatus:'M'},ignore_index=True)  #Added AnnotationStatus as M 
                #logs.createAnnotationsFile(df_userAnnot)
                
                #Remove the selected sample from the original dataframe
                df_predictions.drop(index=indexValue,inplace=True)   
                df_predictions.reset_index(inplace=True,drop=True)
            else:
                print("All of unlabelled data is over")            
                    
                #df_manuallyAnnotated = pd.concat([df_manuallyAnnotated,df_userAnnot])
                
            iteration+=1
        
    # Instead of Stopping and Manually annotating, for the sake of efficiency of research, all labels are already Manullay annotated in dataset
    df_predictions=df_predictions[[req1,req2,label,annStatus]]#df_predictions[['comboId','req1Id','req1',req1,'req2Id','req2',req2,label,annStatus]]
    df_remaining = df_predictions
    df_remaining[annStatus] = ''
    #df_manuallyAnnotated=df_manuallyAnnotated[['comboId','req1Id','req1',req1,'req2Id','req2',req2,label,annStatus]]
    logs.writeLog("\n\nManually Annotated Combinations... "+str(len(df_predictions))+"Rows \n"+str(df_predictions[:10]))

    df_new_laballed = df_userAnnot + df_confident_predictions
    
    return df_new_laballed, df_remaining
