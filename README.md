# fake-news-detection-using-python
in this project we determine the news validity we uses NPL processing in this project \\ this project is made by using references 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# NLP libraries to clean the text data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Vectorization technique TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# For Splitting the dataset
from sklearn.model_selection import train_test_split

# Model libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#Accuracy measuring library
from sklearn.metrics import accuracy_score
# 2. Loading the data
file_path = '../input/fake-news-detection/data.csv'
data = pd.read_csv(file_path)
data.shape #Returns the number of rows and columns present in the dataset
(4009, 4)
data.head()  # Returns the first 5 rows of the dataset
***URLs	Headline	Body	Label
0	http://www.bbc.com/news/world-us-canada-414191...	Four ways Bob Corker skewered Donald Trump	Image copyright Getty Images\nOn Sunday mornin...	1
1	https://www.reuters.com/article/us-filmfestiva...	Linklater's war veteran comedy speaks to moder...	LONDON (Reuters) - “Last Flag Flying”, a comed...	1
2	https://www.nytimes.com/2017/10/09/us/politics...	Trump’s Fight With Corker Jeopardizes His Legi...	The feud broke into public view last week when...	1
3	https://www.reuters.com/article/us-mexico-oil-...	Egypt's Cheiron wins tie-up with Pemex for Mex...	MEXICO CITY (Reuters) - Egypt’s Cheiron Holdin...	1
4	http://www.cnn.com/videos/cnnmoney/2017/10/08/...	Jason Aldean opens 'SNL' with Vegas tribute	Country singer Jason Aldean, who was performin...	1***
data.columns # Returns the column headings
Index(['URLs', 'Headline', 'Body', 'Label'], dtype='object')
data.isnull().sum() #To check the null values in the dataset, if any
**URLs         0
Headline     0
Body        21
Label        0
dtype: int64**
# 3.Data-Preprocessing
df = data.copy() 
# 3.1. Removing the Null Values
df['Body'] = df['Body'].fillna('')   # As Body is empty, just filled with an empty space
df.isnull().sum()  # No null values found
**URLs        0
Headline    0
Body        0
Label       0
dtype: int64**
# 3.2. Adding a new column
df['News'] = df['Headline']+df['Body']
df.head()
**data
URLs	Headline	Body	Label	News
0	http://www.bbc.com/news/world-us-canada-414191...	Four ways Bob Corker skewered Donald Trump	Image copyright Getty Images\nOn Sunday mornin...	1	Four ways Bob Corker skewered Donald TrumpImag...
1	https://www.reuters.com/article/us-filmfestiva...	Linklater's war veteran comedy speaks to moder...	LONDON (Reuters) - “Last Flag Flying”, a comed...	1	Linklater's war veteran comedy speaks to moder...
2	https://www.nytimes.com/2017/10/09/us/politics...	Trump’s Fight With Corker Jeopardizes His Legi...	The feud broke into public view last week when...	1	Trump’s Fight With Corker Jeopardizes His Legi...
3	https://www.reuters.com/article/us-mexico-oil-...	Egypt's Cheiron wins tie-up with Pemex for Mex...	MEXICO CITY (Reuters) - Egypt’s Cheiron Holdin...	1	Egypt's Cheiron wins tie-up with Pemex for Mex...
4	http://www.cnn.com/videos/cnnmoney/2017/10/08/...	Jason Aldean opens 'SNL' with Vegas tribute	Country singer Jason Aldean, who was performin...	1	Jason Aldean opens 'SNL' with Vegas tributeCou...**

df.columns
Index(['URLs', 'Headline', 'Body', 'Label', 'News'], dtype='object')
# 3.3. Drop features that are not needed
features_dropped = ['URLs','Headline','Body']
df = df.drop(features_dropped, axis =1)
df.columns
Index(['Label', 'News'], dtype='object')
# 3.4. Text Processing
Remove symbols(',','-',...etc)
Remove stop words
Stemming
ps = PorterStemmer()
def wordopt(text):
    text = re.sub('[^a-zA-Z]', ' ',text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text
    df['News'] = df['News'].apply(wordopt) #Applying the text processing techniques onto every row data
df.head()
**Label	News
0	1	four way bob corker skewer donald trumpimag co...
1	1	linklat war veteran comedi speak modern americ...
2	1	trump fight corker jeopard legisl agendath feu...
3	1	egypt cheiron win tie pemex mexican onshor oil...
4	1	jason aldean open snl vega tributecountri sing...
**
# Splitting DataSet
X = df['News']
Y = df['Label']

# Split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
# 5. Vectorization
This is used to handle our text data, by converting it into vectors.
 # Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
# 6. Model Fitting I will be fitting my data onto 3 classifications models
# Logistic Regression
# SVM
# RandomForestClassifier
# The best one amongst the 3 will be used further
# 1. Logistic Regression - used because this model is best suited for binary classification
LR_model = LogisticRegression()

# Fitting training set to the model
LR_model.fit(xv_train,y_train)

# Predicting the test set results based on the model
lr_y_pred = LR_model.predict(xv_test)

# Calculate the accurracy of this model
score = accuracy_score(y_test,lr_y_pred)
print('Accuracy of LR model is ', score)
Accuracy of LR model is  0.9720837487537388
#2. Support Vector Machine(SVM) - SVM works relatively well when there is a clear margin of separation between classes.
svm_model = SVC(kernel='linear')

# Fitting training set to the model
svm_model.fit(xv_train,y_train)
# Predicting the test set results based on the model
svm_y_pred = svm_model.predict(xv_test)
# Calculate the accuracy score of this model
score = accuracy_score(y_test,svm_y_pred)
print('Accuracy of SVM model is ', score)
Accuracy of SVM model is  0.9860418743768694

# 3. Random Forest Classifier 
RFC_model = RandomForestClassifier(random_state=0)
# Fitting training set to the model
RFC_model.fit(xv_train, y_train)
# Predicting the test set results based on the model
rfc_y_pred = RFC_model.predict(xv_test)

# Calculate the accuracy score of this model
score = accuracy_score(y_test,rfc_y_pred)
print('Accuracy of RFC model is ', score)
Accuracy of RFC model is  0.9710867397806581
# 7. Manual Model Testing
# As SVM is able to provide best results - SVM will be used to check the news liability

def fake_news_det(news):
    input_data = {"text":[news]}
    new_def_test = pd.DataFrame(input_data)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    #print(new_x_test)
    vectorized_input_data = vectorization.transform(new_x_test)
    prediction = svm_model.predict(vectorized_input_data)
    
    if prediction == 1:
        print("Not a Fake News")
        else:
        print("Fake News")
fake_news_det('U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.')
Not a Fake News
