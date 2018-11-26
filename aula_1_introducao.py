from sklearn.naive_bayes import MultinomialNB
#Definindo instâncias para treino
porco1=[1,1,0]
porco2=[1,1,0]
porco3=[1,1,0]
cachorro1=[1,1,1]
cachorro2=[0,1,1]
cachorro3=[0,1,1]
dados=[porco1,porco2,porco3,cachorro1,cachorro2,cachorro3]
marcacoes=[1,1,1,-1,-1,-1]
#Dataset não classificado
test=[[1,1,1],[1,0,0],[1,0,1]]
#Resultado esperado das classificações
marcacoes_teste=[-1,1,-1]
modelo = MultinomialNB()
modelo.fit(dados, marcacoes)
result=modelo.predict(test)
print(result-marcacoes_teste)