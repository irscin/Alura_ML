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
treino_dados=X[:90]
treino_marcacoes=Y[:90]

teste_dados=X[-9:]
teste_marcacoes=Y[-9:]

modelo.fit(treino_dados,treino_marcacoes)
result=modelo.predict(teste_dados)
#Calculando taxa de acerto
dif=result-teste_marcacoes
acertos=[d for d in dif if d==0]
total_acertos=len(acertos)
total=len(teste_marcacoes)
print(100.0*total_acertos/total)