
# coding: utf-8

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


given_images = pd.read_csv('train_digitalRecogniser.csv')
given_images.head()


# In[5]:


images = given_images.iloc[0:,1:]
images.head()


# In[139]:


images= images.reshape(images.shape[0], 28, 28)
for i in range(0, 3):
    
    plt.subplot(330 + (i+1))
    plt.imshow(images[i], cmap=plt.get_cmap('gist_earth'))
    plt.title(labels[i])
    


# In[127]:


labels = given_images.iloc[0:,0]
labels.tail()


# In[12]:


train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size = 0.75, test_size = 0.25, random_state= 1)


# In[13]:


test_images_final = test_images[test_images > 0]
test_images_final.fillna(1, inplace = True)


train_images_final = train_images[test_images > 0]
train_images_final.fillna(1, inplace = True)
train_images_final.head()



#test_images[test_images>0]=1


# In[41]:


#clf = svm.SVC()
clf = RandomForestClassifier()
#clf = DecisionTreeClassifier()
clf.fit(train_images, train_labels)
label_pred = clf.predict(test_images)


# In[42]:


accuracy = accuracy_score(test_labels, label_pred)
print(accuracy)


# In[77]:


df1 = pd.read_csv('test_digitalRecogniser.csv')
df1
result = clf.predict(df1)
df = pd.DataFrame(result)
df.index.name ='ImageId'
df.index+=1
df.columns = ['Label']
df.to_csv('results_digital recogniser.csv' , header = True)

df.head()
#df.shape

