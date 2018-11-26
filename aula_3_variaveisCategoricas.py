# -*- coding: utf-8 -*-
# import csv
# def load_data():
#     X=[]
#     Y=[]
#     arquivo=open('cursos.csv','rb')
#     leitor=csv.reader(arquivo)
#     leitor.next()
#     for home, busca, logado, comprou in leitor:
#         X.append([int (home), busca, int (logado)])
#         Y.append([int (comprou)])
#     return X, Y

import pandas as pd
from sklearn.naive_bayes import MultinomialNB 
#Lendo e tratando dados
df=pd.read_csv('cursos.csv')
X_df=df[['home', 'busca', 'logado']]
Y_df=df[['comprou']]
#Transformando coluna de variáveis categóricas em várias colunas binárias
X_df=pd.get_dummies(X_df)
#Pegando apenas valores, descartando os nomes das colunas
X_df=X_df.values
Y_df=Y_df.values

tam_treino=0.9*len(X_df)
tam_teste=len(X_df)-tam_treino

X_treino=X_df[:int (tam_treino)]
Y_treino=Y_df[:int (tam_treino)]

X_teste=X_df[int(-tam_teste):]
Y_teste=Y_df[int(-tam_teste):]
#Treinando modelo e predizendo testes
modelo=MultinomialNB()
#Chamando função ravel pois preciso passar um array de arrays e não somente um array para a função fit
modelo.fit(X_treino, Y_treino.ravel())
#Calculando taxa de acerto
resultado=modelo.predict(X_teste)
#Chamando a função flatten para converter um array de arrays em um array simples
dif=resultado-Y_teste.flatten()

acertos=[d for d in dif if d==0]
total_acertos=len(acertos)
total=len(Y_teste)
print(100.0*total_acertos/total)