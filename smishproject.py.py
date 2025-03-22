import pandas as pd
import numpy as np
import re
import pickle
import nltk
nltk.download('stopwords')
#nltk.download('wordnet')
txt_list=[]
from nltk.corpus import stopwords  #corpus,stem-module stop--remove unwanted words
#from nltk.stem import WordNetLemmatizer #lematizing-chanhge the word into simple word 
df=pd.read_csv('smsdataset',header=None,sep='\t',names=['label','text']) #data loadedprint(df.head())
##[print(df.groupby('label').describe())
#preprocess
for i in range(0,len(df)):
    result=re.sub('[^a-zA-Z]',' ',df['text'][i]) #remove non alpabet
    result=result.lower() #lowercase
    result=result.split() #tokenize
    result=' '.join([word for word in result if word not in stopwords.words('english')])
    txt_list.append(result)
#print(txt_list)
#feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer #term frequency inverse document frequency means it state or find the importance of word in a secntence 
Tfidf_vec=TfidfVectorizer() #initialize
x=Tfidf_vec.fit_transform(txt_list).toarray() #input
y=pd.get_dummies(df['label']) #output
with open('Tfidf_vec.pkl','wb') as f:
    pickle.dump(Tfidf_vec,f)
#print(y)
##y=y.iloc(1).values
##print(x)
##print(y)
#split the dataset for train and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
##y_train=y_train.astype(int)
#print(y_train)
#below given because our lstm model require 3 vector 
x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])
x_test=x_test.reshape(x_test.shape[0],1,x_test.shape[1])
#model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,LSTM,Dense,Embedding,Dropout
model=Sequential() #create base model
model.add(Input(shape=(x_train.shape[1],x_train.shape[2])))
model.add(LSTM(units=64,return_sequences=False)) #input layer
model.add(Dropout(0.2)) #20% of neauorn nout in use
model.add(Dense(units=32,activation='relu')) #hidden layer
model.add(Dropout(0.2))
model.add(Dense(units=2,activation='softmax'))
#model compile
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
##model.summary()
#train the model
model_train=model.fit(x_train,y_train,epochs=25,batch_size=64,validation_data=(x_test,y_test))
#model evaluation
loss,accuracy=model.evaluate(x_test,y_test)
#print(f'test accuracy:{accuracy}')
model.save("smish_model.h5")
#model.save("tfidf.h5")
#new input
input="Hey, check out this amazing discount on your next online shopping! You can claim your offer by clicking here: https://bit.ly/discount-offer-now. Don't miss out!"
def preprocess_input(text):
    result_text=re.sub('[^a-zA-Z]',' ',text)
    result_text=result_text.lower()
    result_text=result_text.split()
    result_text=' '.join([word for word in result_text if word not in stopwords.words('english')])
    return result_text
preprocess=preprocess_input(input)
input_vec=Tfidf_vec.transform([preprocess]).toarray()
input_vec=input_vec.reshape(input_vec.shape[0],1,input_vec.shape[1])
#print(input_vec)
prediction=model.predict(input_vec,verbose=0)
#print(prediction)
predict_class=np.argmax(prediction,axis=1)
if predict_class==0:
    print("This is HAM msg")
else:
    print("This is Smish msg")
