from sklearn.naive_bayes import MultinomialNB
porco1=[1,1,0]
porco2=[1,1,0]
porco3=[1,1,0]
cachorro1=[1,1,1]
cachorro2=[0,1,1]
cachorro3=[0,1,1]
dados=[porco1,porco2,porco3,cachorro1,cachorro2,cachorro3]
marcacoes=[1,1,1,-1,-1,-1]
test=[[1,1,1],[1,0,0]]
modelo = MultinomialNB()
modelo.fit(dados, marcacoes)
print(modelo.predict(test))