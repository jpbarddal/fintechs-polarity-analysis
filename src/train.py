# coding: utf-8
import nltk
import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
# nltk.download('all')

pathAtual = "./"


#### reads the file
path = "../data/"
datasetName = 'dataset.csv'
stopwordsName = 'customstopwords.txt'

rows = []
with open(path + datasetName, 'r') as simple:
    trainFile = csv.reader(simple)
    for row in trainFile:
       tblb = ()
       tblb = (row[0], row[1])
       rows.append(tblb)

#### stopwords removal

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
with open(path + stopwordsName, 'r') as simple:
    file = csv.reader(simple)
    for row in file:
       stopwordsnltk.append(row[0])

#Remove stopWords ==> para ver como funciona a extração mesmo vai ocorrer na função aplicastemmer
def removestopwords(content):
    sentences = []
    for (words, emotion) in content:
        #split serve para quebrar as palavras, ele acha um espaco em branco e quebra - passar o stopwordsnltk para usar a lista do nltk.
        #print(emocao)
        semstop = [p for p in words.split() if p not in stopwordsnltk]

        #adicionar dentro da variavel frases - passa como parametro emocao para manter a classe
        sentences.append((semstop, emotion))
    return sentences

#aplica stemming
def stemmer(texto):
    # criar a variavel stemmer que vai receber como parametro o pacote nltk.stem que tem varios tipos de stemmers, o melhor é RSLPSStemmer
    # que é especifico para lingua portuguesa.
    stemmer = nltk.stem.RSLPStemmer()
    # criar uma variavel frasesstemming que sera um vetor, lista que tera todas as palavras já retirando o radical e stopwords
    frasessstemming = []
    # primeira variavel pega as palavras do texto e a segunda pegará a classe a que a frase pertence.
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasessstemming.append((comstemming, emocao))
    return frasessstemming

#### runs the stemming process
stemmingSentences = stemmer(rows)


def buscapalavras(frases):
    todaspalavras = []
    for (palavras, polaridade) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

#criar uma variavel palavra chamamos a funcao e passamos como parametro as frases com stemming e trazer a lista simples com todas as palavras e
#valores repetidos. Esses valores nao poderao ser repetidos. Tem que fazer funcao para extrair as palavras unicas da base de dados.
palavrastreinamento = buscapalavras(stemmingSentences)

#devemos extrair as palavras unicas da base de dados e com isso eliminar as repetidas.
#para isto usamos a função que busca a frequencia das palavras (quantas vezes aparecem na base) e depois pegamos só a palavra
#usa como parametro a lista de palavras obtida na função anterior
def buscafrequencia(palavras):
    #verifica quantas vezes a palavra aparece na base
    palavras = nltk.FreqDist(palavras)
    return palavras

frequenciatreinamento = buscafrequencia(palavrastreinamento)


def buscapalavrasunicas(frequencia):
    #elimina a repetição de palavras - pega só a palavra assim uma palavra que aparece 4 vezes (am:4) será reduzida para só uma eliminando as repetições
    freq = frequencia.keys()
    return freq

palavrasunicastreinamento = buscapalavrasunicas(frequenciatreinamento)


#ver que palavras existem em cada frase da base
def veSeTemPalavra(frase):
    fra = set(frase)
    caracteristicas = []
    for palavras in palavrasunicastreinamento:
        v = 0
        if palavras in fra:
            v = 1
        caracteristicas.append(v)
    return caracteristicas

basecompletatreinamento = nltk.classify.apply_features(veSeTemPalavra, stemmingSentences)



set = list(basecompletatreinamento)
X = np.asarray([i[0] for i in set])
y = np.asarray([i[1] for i in set])

sss = StratifiedKFold(n_splits=10)

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

accsNB = []
accsRF = []
accsDT = []
for train_index, test_index in sss.split(X, y):
    xTrain, yTrain = X[train_index], y[train_index]
    xTest, yTest   = X[test_index],  y[test_index]

    ### Naive Bayes
    nb = MultinomialNB()
    nb.fit(xTrain, yTrain)
    accsNB.append(accuracy_score(yTest, nb.predict(xTest)))

    ### Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(xTrain, yTrain)
    accsDT.append(accuracy_score(yTest, dt.predict(xTest)))

    ### Random Forest
    rf = RandomForestClassifier()
    rf.fit(xTrain, yTrain)
    accsRF.append(accuracy_score(yTest, rf.predict(xTest)))

avgNB = np.mean(accsNB)
avgDT = np.mean(accsDT)
avgRF = np.mean(accsRF)
print("Avg. Acc obtained for NB classifier = {}".format(avgNB))
print("Avg. Acc obtained for DT classifier = {}".format(avgDT))

print("Avg. Acc obtained for RF classifier = {}".format(avgRF))