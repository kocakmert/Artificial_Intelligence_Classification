# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:06:02 2021

@author: kocak
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler # ölçeklendirme 
from sklearn.metrics import confusion_matrix # confisuon matris  kütüphane
from sklearn.metrics import plot_confusion_matrix #confusion matris çizim kütüphane
from sklearn.metrics import classification_report  #değerlendirme kriterleri kütüphane
#%%
# data işlemleri
data = pd.read_csv("data.csv")
print(data.shape)
print(len(data))
data.head()
#%%

# 1- hastalıklı ve sağlıklı sayıları
sns.set_theme(style="darkgrid")
sns.countplot(x="target",data=data,palette="Set3")
#%%
# 2- Cinsiyete göre hasta ve sağlıklı  sayıları
hasta_erkek=0
sagliklı_erkek=0
hasta_kadin=0
saglikli_kadin=0
x=data[['sex','target']].values
#print(x)
a=0
for i in range(0,x.shape[0]):
    if(x[i][a]==0 and x[i][a+1] == 0):
        saglikli_kadin +=1
    elif(x[i][a]==0 and x[i][a+1] == 1):
        hasta_kadin +=1
    elif(x[i][a]==1 and x[i][a+1] == 0):
        sagliklı_erkek+=1
    elif(x[i][a]==1 and x[i][a+1] == 1):
        hasta_erkek+=1          
#print(saglikli_kadin)
#print(hasta_kadin)
#print(hasta_kadin)

barWidth = 0.3
bars1 = [sagliklı_erkek,saglikli_kadin]  #mavi bar sağlıklıklı bireyler
bars2 = [hasta_erkek,hasta_kadin]  # cyan hasta bireyler
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='Sağlıklı')
plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='Hasta')

plt.xticks([r + barWidth for r in range(len(bars1))], ['Erkek', 'Kadın'])
plt.ylabel('Sayısı')
plt.legend()
plt.show()

#%%
# 3- Veri Setindekilerin  Yaş dağılımı
sns.histplot(data["age"], kde=True)

#%%
# 4 - ...  Hasta olanların yaş dağılımı ... 
hastalar=data[['age','target']].values
hasta_age= []
s=0
for t in range(0,len(hastalar)):
    if(hastalar[t][1] == 1):
        hasta_age.append(hastalar[t][0])
print(hasta_age)

sns.histplot(hasta_age, kde=True)

#%%
# 5 - ... Veri Set'inin Sınıflandırma İçin Hazırlanması ...
x_data=data.drop(["target"],axis=1)
y_data=data['target'].values

# --- Ölçeklendirme ---  #Logistic Regresyonda Ölçeklendirme yapılmadığında  daha yüksek oranlar çıkmakta...
sc_x=StandardScaler()
x_data=sc_x.fit_transform(x_data)

#print(x_data)
#%%
# 6 - ... Veri Set'inin Eğitim ve Test Olarak Ayrılması ...
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=42)

#%%

# 7   --- Logistic Regresyon ---

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(max_iter=100)

logmodel.fit(x_train,y_train) #veri modele fit edilmesi.(θ ları bulacak)

logmodel.coef_ # θ'lar 

pred=logmodel.predict(x_test)  #tahmin 
#x_test.shape
#print(pred)
#confusion matrix
confusion_matrix(y_test,pred)

#confusion matrix çizim 
plot_confusion_matrix(logmodel,x_test,y_test,display_labels=["(0)Sağlıklı","(1) Hasta"],cmap=plt.cm.Blues)

#değerlendirme kriterleri
print(classification_report(y_test,pred))

## --- Değerlendirme Kriterleri Hesaplama --- Kütüphanesiz
matrix=confusion_matrix(y_test,pred)

TP=matrix[0][0]  #TRUE POSİTİVE
FP=matrix[0][1]  #FALSE POSİTİVE
FN=matrix[1][0]  #FALSE NEGATİF
TN=matrix[1][1]  #TRUE NEGATİVE
sum_data=x_test.shape[0]  #TOPLAM

# Doğrululuk
Accuracy=(TP+TN) / sum_data
print("Doğruluk Oranı : "+ str(Accuracy))

#Duyarlılık
Sensitivity= TP / (TP+FN)
print("Duyarlılık Oranı : "+ str(Sensitivity))

#Özgüllük
Specificity= TN / (TN+FP)
print("Özgüllük Oranı: "+ str(Specificity))
#%%

# --- 8-  K-NN (k neirest neighbour) ---

#--- Ölçeklendirme ---
sc_x=StandardScaler()
x_data=sc_x.fit_transform(x_data)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=42)
 
from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=15,p=2,metric='euclidean')
classifier.fit(x_train,y_train)

y_pred_KNN=classifier.predict(x_test)  #predicition
#print(y_pred)
#confusion matrix
confusion_matrix(y_test,y_pred_KNN)

#confusion matrix çizim 
plot_confusion_matrix(classifier,x_test,y_test,display_labels=["(0)Sağlıklı","(1) Hasta"],cmap=plt.cm.Oranges)

#değerlendirme kriterleri
print(classification_report(y_test,y_pred_KNN))

#------------------------------------------------------------------------------------------------------------------------
## --- Değerlendirme Kriterleri Hesaplama --- Kütüphanesiz
matrix_KNN=confusion_matrix(y_test,y_pred_KNN)

TP=matrix_KNN[0][0]  #TRUE POSİTİVE
FP=matrix_KNN[0][1]  #FALSE POSİTİVE
FN=matrix_KNN[1][0]  #FALSE NEGATİF
TN=matrix_KNN[1][1]  #TRUE NEGATİVE
sum_data=x_test.shape[0]  #TOPLAM

# Doğrululuk
Accuracy=(TP+TN) / sum_data
print("Doğruluk Oranı : "+ str(Accuracy))

#Duyarlılık
Sensitivity= TP / (TP+FN)
print("Duyarlılık Oranı : "+ str(Sensitivity))

#Özgüllük
Specificity= TN / (TN+FP)
print("Özgüllük Oranı: "+ str(Specificity))

#%%

# 9-  ---  Naive Bayes ---

# --- Ölçeklendirme ---
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

from sklearn.naive_bayes import GaussianNB

nvclassifier = GaussianNB()
nvclassifier.fit(x_train,y_train)

y_pred_naivebayes= nvclassifier.predict(x_test)

#confusion matrix 
confusion_matrix(y_test,y_pred_naivebayes)

#confusion matrix çizim 
plot_confusion_matrix(nvclassifier,x_test,y_test,display_labels=["(0)Sağlıklı","(1) Hasta"],cmap=plt.cm.Blues)

#değerlendirme kriterleri
print(classification_report(y_test,y_pred_naivebayes))

#------------------------------------------------------------------------------------------------------------------------
## --- Değerlendirme Kriterleri Hesaplama 
matrix_Naive_Bayes=confusion_matrix(y_test,y_pred_naivebayes)

TP=matrix_Naive_Bayes[0][0]  #TRUE POSİTİVE
FP=matrix_Naive_Bayes[0][1]  #FALSE POSİTİVE
FN=matrix_Naive_Bayes[1][0]  #FALSE NEGATİF
TN=matrix_Naive_Bayes[1][1]  #TRUE NEGATİVE
sum_data=x_test.shape[0]  #TOPLAM

# Doğrululuk
Accuracy=(TP+TN) / sum_data
print("Doğruluk Oranı : "+ str(Accuracy))

#Duyarlılık
Sensitivity= TP / (TP+FN)
print("Duyarlılık Oranı : "+ str(Sensitivity))

#Özgüllük
Specificity= TN / (TN+FP)
print("Özgüllük Oranı: "+ str(Specificity))

#%% 

# 10-  --- Decision Tree (Karar Ağaçları) --- 

# --- Ölçeklendirme ---
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

from sklearn.tree import DecisionTreeClassifier

dtclassifier=DecisionTreeClassifier()
dtclassifier.fit(x_train,y_train)

y_pred_decisionTree= dtclassifier.predict(x_test)

#confusion matrix 
cm_dt=confusion_matrix(y_test,y_pred_decisionTree)
print(cm_dt)

#confusion matrix çizim 
plot_confusion_matrix(dtclassifier,x_test,y_test,display_labels=["(0)Sağlıklı","(1) Hasta"],cmap=plt.cm.Blues)

#değerlendirme kriterleri
print(classification_report(y_test,y_pred_decisionTree))

matrix_Decision_Tree=confusion_matrix(y_test,y_pred_decisionTree)

TP=matrix_Decision_Tree[0][0]  #TRUE POSİTİVE
FP=matrix_Decision_Tree[0][1]  #FALSE POSİTİVE
FN=matrix_Decision_Tree[1][0]  #FALSE NEGATİF
TN=matrix_Decision_Tree[1][1]  #TRUE NEGATİVE
sum_data=x_test.shape[0]  #TOPLAM
print(x_test.shape)
# Doğrululuk
Accuracy=(TP+TN) / sum_data
print("Doğruluk Oranı : "+ str(Accuracy))

#Duyarlılık
Sensitivity= TP / (TP+FN)
print("Duyarlılık Oranı : "+ str(Sensitivity))

#Özgüllük
Specificity= TN / (TN+FP)
print("Özgüllük Oranı: "+ str(Specificity))

#%%

# 11- ---  Artificial neural networks (Yapay Sinir Ağları) ---

# --- Ölçeklendirme ---
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

from keras.models import Sequential#For building the Neural Network layer by layer
from keras.layers import Dense#build our layers library

classifier_ANN = Sequential() # İNİTİALİZE NEURAL NETWORK
classifier_ANN.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
classifier_ANN.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) # Adding the second hidden layer
classifier_ANN.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) # Adding the output layer
# Compiling the ANN
classifier_ANN.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  #'adam'=adaptive momentum (learning rate sabit olmuyor...)
classifier_ANN.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)# Fitting the ANN to the Training set
y_pred_ANN = classifier_ANN.predict(x_test)

print(y_pred_ANN)
print(y_pred_ANN.shape)

for i in range(0, y_pred_ANN.shape[0]):
    if(y_pred_ANN[i][0] > 0.5):
        y_pred_ANN[i][0] = 1
    else:
        y_pred_ANN[i][0] =0
print(y_pred_ANN)

#confusion matrix 
cm_ann = confusion_matrix(y_test, y_pred_ANN)
print(cm_ann)

#değerlendirme kriterleri

print(classification_report(y_test,y_pred_ANN))

matrix_ANN=confusion_matrix(y_test,y_pred_ANN)

TP=matrix_ANN[0][0]  #TRUE POSİTİVE
FP=matrix_ANN[0][1]  #FALSE POSİTİVE-
FN=matrix_ANN[1][0]  #FALSE NEGATİF
TN=matrix_ANN[1][1]  #TRUE NEGATİVE
sum_data=x_test.shape[0]  #TOPLAM
print(x_test.shape)
# Doğrululuk
Accuracy=(TP+TN) / sum_data
print("Doğruluk Oranı : "+ str(Accuracy))

#Duyarlılık
Sensitivity= TP / (TP+FN)
print("Duyarlılık Oranı : "+ str(Sensitivity))

#Özgüllük
Specificity= TN / (TN+FP)
print("Özgüllük Oranı: "+ str(Specificity))




