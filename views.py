from django.shortcuts import render
from complain_app.models import Complain
import pymongo
import pandas, xgboost, numpy, textblob, string
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
# Create your views here.
from django.http import HttpResponse

classifier = naive_bayes.MultinomialNB()
#def complain_app(request):
    
myclient=pymongo.MongoClient("mongodb://localhost:27017/")
mydb=myclient["project_db"]
mycol=mydb["Department_Classification"]

    

labels_train =[]
texts_train=[]
labels_valid=[]
texts_valid=[]
    
for doc in  mycol.find({},{'Department':1,'Word':1}):        
    labels_train.append(doc['Department'])
    texts_train.append(doc['Word'])

    
mycol=mydb["test_data"]
for doc in  mycol.find({},{'Department':1,'message':1}):        
    labels_valid.append(doc['Department'])
    texts_valid.append(doc['message'])
    
    #print(texts_valid)

    #print(texts_train)
    
    #print(len(labels_train))

    #print(texts_valid)
    
     
    
# create a dataframe using texts and lables
trainDF_training = pandas.DataFrame()
trainDF_training['label'] = labels_train
trainDF_training['text'] = texts_train
validDF_testing = pandas.DataFrame()
validDF_testing['label'] = labels_valid
validDF_testing['text'] = texts_valid
    #print(trainDF_training['text'])
    #print(trainDF_training['label'])
    #print(validDF_testing['text'])
    #print(validDF_testing['label'])
# label encode the target variable
train_x = trainDF_training['text']
train_y = trainDF_training['label']

    #print(train_y)
    
valid_x = validDF_testing['text']
valid_y = validDF_testing['label']
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y.values.tolist())
encoder_valid = preprocessing.LabelEncoder()
    #train_y=encoder.fit_transform(train_y)
    #print(train_y.values.tolist())
valid_y = encoder_valid.fit_transform(valid_y.values.tolist())
rev_test=  encoder_valid.inverse_transform(valid_y)
rev_train = encoder.inverse_transform(train_y)
    
    #print(train_y)
    #print(valid_y)
    #print(rev_test)


    
count_vect_train = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# following command put each word in a matrics in alphabetic order
count_vect_train.fit(trainDF_training['text'])
count_vect_train.fit(validDF_testing['text'])
    #print(count_vect_train.vocabulary)
    #print(count_vect_train.get_feature_names())
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect_train.transform(train_x)
xvalid_count =  count_vect_train.transform(valid_x)
    #print(xtrain_count.shape)
    #print(xtrain_count)
    #print(xvalid_count.shape)



    
    # fit the training dataset on the classifier
    #classifier = naive_bayes.MultinomialNB()
def complain_training(request):
    if request.method == 'POST':

        classifier.fit(xtrain_count, train_y)
        
        #save the model
        filename = 'finalized_model.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        #print("heloo hiiiii")
        
        

       
        #load the model
        #loaded_model = pickle.load(open(filename, 'rb'))
        #print(loaded_model)
        '''predictions = classifier.predict(xvalid_count)
        print(predictions) 
        accuracy = loaded_model.score(xvalid_count, valid_y)
        print(accuracy)'''
            
            #rev = encoder1.inverse_transform(predictions)
        
        
        #return HttpResponse()
        return render(request,'buttons_train.html')
    else:
        #return HttpResponse()
        return render(request,'buttons_train.html')
def complain_prediction(request):
    if request.method == 'POST':
        filename = 'finalized_model.sav'
        
        loaded_model = pickle.load(open(filename, 'rb'))
        predictions = loaded_model.predict(xvalid_count)
        rev = encoder_valid.inverse_transform(predictions)
        print(rev)
        #print(predictions) 
        accuracy = loaded_model.score(xvalid_count, valid_y)
        #print(accuracy)
        #for i in range()
        #_id[i] = mycol.find({})
        
        cursor_list = []
        
        cur = mycol.find({},{'_id':1})
        for j in range(len(rev)):
            cursor_list.append(cur.next())
        id_list = [ i['_id'] for i in cursor_list ]
        #print(id_list[0])
            
        #print(cursor_list)
        for i in range(len(rev)):
            #print(cursor_list[i])
            #print (i, end = " ") 
            #print (rev[i])
            mycol.update_one({"_id":id_list[i]},{'$set':{"result":rev[i]}})
            

        
        
        return render(request,'buttons_train.html')
    else:
        #return HttpResponse()
        return render(request,'buttons_train.html')









