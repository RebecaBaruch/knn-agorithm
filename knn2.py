import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 

redWineData = pd.read_csv('./database/wine.csv',header=None)
print(redWineData)

# verifica se ainda há dados no DataFrame
if redWineData.empty:
    print("O DataFrame está vazio após a remoção de valores ausentes.")
else:
    # criar uma matriz X e o vetor y
    x = np.array(redWineData.iloc[:, 1:14]) # features
    y = np.array(redWineData.iloc[:, 0]) # classes

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
        


#Com a base não normalizada a acurácia foi menor. Com a base normalizada, a acurácia foi maior. O KNN utiliza o cálculo da distância entre os atributos, quanto maior a distância mais difícil de se calcular a acurácia