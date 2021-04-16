# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.model_selection import train_test_split   #nos permite dividir a entrada em grupos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler #padroniza variaveis de entrada para uma mesma escala
from sklearn.metrics import confusion_matrix   #gera a matriz confusão
from sklearn.metrics import accuracy_score     
from sklearn.datasets import load_iris
import numpy as np
from sklearn.neural_network import MLPClassifier #Multi-layer Perceptron classifier.


#Carrega o iris dataset em iris 
iris = load_iris()
X = iris.data
y = iris.target 


#divisao do data set entra grupo de treino e test
cetosa= iris.data[:50,:]
cetosay= iris.target[:50]
cet=np.array(cetosa)
cety=np.array(cetosay)
cettrain, cettest, cetytrain, cetytest=train_test_split(cet, cety, random_state=3)

versicolor= iris.data[50:100,:]
versicolory= iris.target[50:100]
ver=np.array(versicolor)
very=np.array(versicolory)
vertrain, vertest, verytrain, verytest=train_test_split(ver, very, random_state=3)

virginica= iris.data[100:150,:]
virginicay= iris.target[100:150]
virg=np.array(virginica)
virgy=np.array(virginicay)
virgtrain, virgtest, virgytrain, virgytest=train_test_split(virg, virgy, random_state=3)

Xtest = np.concatenate((cettest, vertest, virgtest), axis=0)
Xtrain = np.concatenate((cettrain, vertrain, virgtrain), axis=0)
Ytest = np.concatenate((cetytest, verytest, virgytest), axis=0)
Ytrain = np.concatenate((cetytrain, verytrain, virgytrain), axis=0)

#Padronizando data set, necessario para correto funcionamento do algoritimo NN (z = (x - u) / s).
scaler = StandardScaler()
scaler.fit(Xtrain)

Xtest2 = scaler.transform(Xtest)        #variáveis de treino e test Padronizadas
Xtrain2 = scaler.transform(Xtrain)


#criando o modelo do clasificador usando Multi-layer Perceptron classifier

mlp = MLPClassifier(hidden_layer_sizes= ( 16 , 16 ) , max_iter = 1500)    #usando Adam

# mlp = MLPClassifier(hidden_layer_sizes=(12, 12) , solver = 'lbfgs', max_iter=1500)  #usando outro otimizador 

"""
    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed
          by Kingma, Diederik, and Jimmy Ba

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.
"""

#Treinamento do modelo
mlp.fit(Xtrain2, Ytrain)

#Previsão dos targets
Y = mlp.predict(Xtest2)
Yt = mlp.predict(Xtrain2)



# Resultados:
    
print('\n Acurácia para o grupo de Teste:')
print(accuracy_score(Y, Ytest))
print('\n Acurácia para o grupo de Treino:')
print(accuracy_score(Yt, Ytrain ))
print('\n Matrix de confusão grupo de Test:')
print(confusion_matrix(Y, Ytest))
print('\n Matrix de confusão grupo de Treino:')
print(confusion_matrix(Yt, Ytrain))
print('\n Relatorio da Classificação:')
print(classification_report(Y, Ytest))