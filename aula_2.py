# -*- coding: utf-8 -*-
import csv
from sklearn.naive_bayes import MultinomialNB
# Funçao de leitura dos dados a partir do CSV
def load_data():
    X=[]
    Y=[]
    arquivo=open('acesso.csv','rb')
    leitor=csv.reader(arquivo)
    leitor.next()
    for home,busca,logado, comprou in leitor:
        X.append([int (home),int (busca),int (logado)])
        Y.append(int (comprou))
    return X, Y

X,Y=load_data()
modelo=MultinomialNB()
treino_dados=X[0:int(len(X)*0.9)]
treino_testes=Y[0:int(len(Y)*0.9)]
modelo.fit(treino_dados,treino_testes)
marcacoes=X[-int(len(X)*0.1):]
marcacoes_teste=Y[-int(len(Y)*0.1):]
result=modelo.predict(marcacoes)
#Calculando taxa de acerto
dif=result-marcacoes_teste
acertos=[d for d in dif if d==0]
total_acertos=len(acertos)
total=len(marcacoes_teste)
print(100.0*total_acertos/total)