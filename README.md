Glassmorphism design
Image result for glassmorphism css design
Glassmorphism is a term used to describe UI design that emphasises light or dark objects, placed on top of colourful backgrounds. A background-blur is placed on the objects which allows the background to shine through – giving it the impression of frosted glass


//cnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import fashion_mnist
import tensorflow.keras as tk

get_ipython().run_line_magic('matplotlib', 'inline')

In[3]:
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
::::::In[4]
display(x_train.shape,x_test.shape)
In[5]:
figure=plt.figure(figsize=(20,20))
for i in range(1,200):
  plt.subplot(20,10,i)
  plt.imshow(x_train[i],cmap=plt.get_cmap('BrBG_r'))
plt.show()
In[12]:
cnn_model = tk.Sequential()
 
cnn_model.add(tk.layers.Conv2D(32,3,3,input_shape = (28,28,1),activation = 'relu'))
 
    # Max pooling will reduce the
    # size with a kernal size of 2x2
cnn_model.add(tk.layers.MaxPooling2D(pool_size= (2,2)))

    # Once the convolutional and pooling
    # operations are done the layer
    # is flattened and fully connected layers
    # are added
cnn_model.add(tk.layers.Flatten())
    
cnn_model.add(tk.layers.Dense(32,activation = 'relu'))
cnn_model.add(tk.layers.Dense(10,activation = 'softmax'))   
In[13]:
cnn_model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
In[14]:
cnn_model.fit(x=x_train,y=y_train,batch_size =512,epochs = 50,verbose = 1,validation_data = (x_test,y_test))
In[15]:
evaluation = cnn_model.evaluate(x_test,y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
In[16]:
predicted_classes = np.argmax(cnn_model.predict(x_test),axis=-1)
In[17]:
predicted_classes
In[50]:
L = 10
W = 10
fig,axes = plt.subplots(L,W,figsize = (20,20))
axes = axes.ravel()
for i in np.arange(0,L*W):
    axes[i].imshow(x_test[i].reshape(28,28))
    axes[i].set_title('Prediction Class:{1} \n true class: {1}'.format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace = 0.75)  
In[38]:
In[31]:
In[ ]:


//
#!/usr/bin/env python
# coding: utf-8
In[45]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
In[36]:
#df1=pd.read_csv("IMDB Dataset.csv")
#df1=df.head(1000)
#df1
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My Drive/Colab Notebooks/
df1=pd.read_csv("IMDB Dataset.csv")
df1.head(1000)
#df1=pd.read_csv("IMDB Dataset.csv")
#df1=df.head(1000)
In[37]:
import tensorflow.keras as tk
In[38]:
from keras.preprocessing.text import Tokenizer
In[39]:
tokenizer1=Tokenizer(oov_token='<nothing>')
In[40]:
tokenizer1.fit_on_texts(df1['review'])
In[41]:
tokenizer1.word_index
In[42]:
tokenizer1.word_counts
In[43]:
tokenizer1.document_count
In[44]:
df1['review']=tokenizer1.texts_to_sequences(df1['review'])
In[46]:
from keras.utils import pad_sequences
seq_df=pd.DataFrame(pad_sequences(df1['review'],padding="post"))
In[47]:
df1 = df1.join(seq_df)
In[48]:
df1.drop(['review'],axis=1,inplace=True)
In[49]:
df1['sentiment'].value_counts()
In[50]:
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['sentiment']=le.fit_transform(df1['sentiment'])
df1['sentiment']
In[51]:
df1.head(2)
In[52]:
from sklearn.model_selection import train_test_split
In[53]:
y = df1['sentiment']
x = df1.iloc[:,1:]
In[54]:
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
In[55]:
import tensorflow.keras as tk
In[56]:
model = tk.Sequential()
model.add(tk.layers.Input(shape=(2493,)))
model.add(tk.layers.Dense(50, activation='relu',kernel_initializer="he_uniform"))
model.add(tk.layers.Dense(1, activation='sigmoid',kernel_initializer="he_uniform"))
model.summary()
In[57]:
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
In[58]:
obj1=model.fit(x=x_train,y=y_train,epochs=50,batch_size=64,validation_data=(x_test,y_test))
In[59]:
y_pred=model.predict(x_test)
In[60]:
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred.round())
In[61]:
accuracy
In[ ]:


// google price

#!/usr/bin/env python
# coding: utf-8
In[1]:
mport Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
In[2]:
read Dataset
df_train=pd.read_csv("Google_Stock_Price_Train.csv")
df_train.head(10)
In[3]:
eras only takes numpy array<br>
ill use Open price for prediction so we need to make it NumPy array
training_set = df_train.iloc[:, 1: 2].values
training_set
In[4]:
cale the stock prices between (0, 1) to avoid intensive computation.
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
training_set=sc.fit_transform(training_set)
training_set
In[5]:
x_train= training_set[0:1257]
y_train= training_set[1:1258]
display(x_train.shape,  y_train.shape)
In[6]:
x_train=np.reshape(x_train, (1257 , 1 , 1))
In[7]:
x_train.shape
In[8]:
df_test=pd.read_csv("Google_Stock_Price_Test.csv")
df_test
In[9]:
figure=plt.figure(figsize=(10,10))
plt.subplots_adjust(top=1.35, bottom=1.2)
df_train['Open'].plot()
plt.ylabel('Open')
plt.xlabel(None)
plt.title(f"Sales Open")
In[10]:
testing_set = df_test.iloc[:, 1: 2].values
testing_set
In[11]:
testing_set=sc.fit_transform(testing_set)
testing_set.shape
In[17]:
x_test= testing_set[0:20]
y_test= testing_set[0:20]
#display(x_test,  y_test)
y_test.shape
In[18]:
x_test=np.reshape(x_test, (20 , 1 , 1))
In[19]:
x_test.shape
In[20]:
import tensorflow.keras as tk
In[32]:
model = tk.Sequential()
model.add(tk.layers.LSTM(units=5, activation= 'sigmoid', input_shape= (None,1)))
model.add(tk.layers.Dense( units=1 ))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=50,validation_data=(x_test,y_test))
In[33]:
y_pred=model.predict(x_test)
In[34]:
plt.plot( y_test , color = 'red' , label = 'Real Google Stock Price')
plt.plot( y_pred , color = 'blue' , label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel( 'time' )
plt.ylabel( 'Google Stock Price' )
plt.legend()
plt.show()
In[ ]:


//boston data
 '''
Problem Statement:-Linear regression by using Deep Neural network:
Implement Boston housing price prediction problem by Linear regression 
using Deep Neural network. Use Boston House price prediction dataset.
'''

# Import Libraries
from sklearn import datasets
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
#ignore warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
# read data from sklearn data set
df=pd.read_csv("Boston.csv")
#df=pd.DataFrame(data.data,columns=data.feature_names)
#df['price']=data.target.
df.rename(columns = {'medv':'price'}, inplace = True)
df
df.info()
df.isnull().sum()
df.describe()
df.shape
#univariate EDA
fig=plt.figure(figsize=(10,10))
df.boxplot()


sns.boxplot(df["rm"])
plt.hist(df["rm"])
df["rm"].value_counts()
#bivariate EDA
sns.scatterplot(x= df["lstat"],y = df["price"])
sns.scatterplot(x= df["rm"],y = df["price"])
#Multivariate EDA
fig= plt.subplots(figsize=(10,10)) 
sns.heatmap(df.corr(),annot=True,cmap="Blues")
pip install keras_tuner
import tensorflow.keras as tk
model=tk.Sequential()
#adding input layer
model.add(tk.layers.Input(shape=(14,)))
#adding first hidden layer
model.add(tk.layers.Dense(units=6,activation="relu",kernel_initializer="he_uniform"))

#adding second hidden layer
model.add(tk.layers.Dense(units=6,activation="relu",kernel_initializer="he_uniform"))
#adding output layer
model.add(tk.layers.Dense(units=6,activation="relu",kernel_initializer="he_uniform"))
#compiling the model
model.compile(optimizer="adam",loss="mean_squared_error",metrics="accuracy")
#compiling the model
model.compile(optimizer="adam",loss="mean_absolute_error")
model.summary()
df.head()
x=df.iloc[:,:-1]
display(x)
y=df['price']
display(y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=10)
#training the model
import time
start=time.time()
obj1=model.fit(x=xtrain,y=ytrain,epochs=50,batch_size=64,validation_data=(xtest,ytest))
