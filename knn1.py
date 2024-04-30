import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 

# lê o arquivo CSV
phisingData = pd.read_csv('./database/phising-websites.csv')
print(phisingData)

# verifica se ainda há dados no DataFrame
if phisingData.empty:
    print("O DataFrame está vazio após a remoção de valores ausentes.")
else:
    # criar uma matriz X e o vetor y
    x = np.array(phisingData.iloc[:, 0:30]) # features
    y = np.array(phisingData['Result']) # classes

    # divide os dados em conjunto de treinamento e conjunto de teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    neighbor = 3
    n = 1

    while n <= 3:
        
        # instancia o classificador KNN
        knn = KNeighborsClassifier(neighbor)

        # treina o modelo
        knn.fit(X_train, y_train)

        # faz previsões no conjunto de teste
        predictions = knn.predict(X_test)

        # calcula a acurácia do modelo
        accuracy = accuracy_score(y_test, predictions) * 100
        print("Valor de K = %.0f" % neighbor)
        print("A acurácia foi de %.2f%%" % accuracy)
        print("_______________________________________")
        
        neighbor += 2
        n += 1
        