# coding: utf-8
import nltk
import pprint
import csv
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC

# nltk.download()

parts = ["juntoEmisturadoModelo.csv"]# "part0.csv", "part1.csv"]

#================================================
#base de dados utilizada para treinar o algoritmo
#================================================
lines = []
for path in parts:
    with open(path, 'r') as simple:
        file = csv.reader(simple)
        for row in file:
           tblb = ()
           tblb = (row[0], row[1])
           lines.append(tblb)

df = pd.DataFrame(lines, columns = ["text", "label"])

#remocao das stopswords
#uso de stopwords do nltk - ja possui recursos para trabalhar com acentuacao em portugues
stopwordsnltk = nltk.corpus.stopwords.words('portuguese')

#Ajusta stopwords acrescentando novas palavras identificadas
ajustaStopWords = 'minhasStopWords.txt'
with open(ajustaStopWords, 'r') as simple:
    ArqStopWords = csv.reader(simple)
    for registro in ArqStopWords:
       stopwordsnltk.append(registro[0])


#aplica stemming
def aplicastemmer(texto):
    # criar a variavel stemmer que vai receber como parametro o pacote nltk.stem que tem varios tipos de stemmers, o melhor é RSLPSStemmer
    # que é especifico para lingua portuguesa.
    stemmer = nltk.stem.RSLPStemmer()
    str = ""
    for p in texto.split():
        if p not in stopwordsnltk:
            # print p
            str = str + " " + re.sub("[^a-zA-Z]]", " ", stemmer.stem(p))
    return str

### cleans the text we have in the dataframe
df['cleaned'] = df['text'].apply(aplicastemmer)


def buscafrequencia(palavras):
    #verifica quantas vezes a palavra aparece na base
    palavras = nltk.FreqDist(palavras)
    return palavras

def buscapalavras(frases):
    todaspalavras = []
    for (palavras, polaridade) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

def buscapalavrasunicas(frequencia):
    #elimina a repetição de palavras - pega só a palavra assim uma palavra que aparece 4 vezes (am:4) será reduzida para só uma eliminando as repetições
    freq = frequencia.keys()
    return freq

# def veSeTemPalavra(frase):
#     fra = set(frase)
#     caracteristicas = {}
#     for palavras in palavrasunicas:
#         caracteristicas['%s' % palavras] = (palavras in fra)
#     return caracteristicas

# frequencia = buscafrequencia(df['cleaned'])
# palavrasunicas = buscapalavrasunicas(frequencia)
# df['cleaned2'] = palavrasunicas
# df['cleaned3'] = df['cleaned2'].apply(veSeTemPalavra)

#### Stratified shuffle split validation

X = df['cleaned']
y = df['label']

sss = StratifiedKFold(n_splits=10)


pipelines = []
params_grid = []
learners = [BernoulliNB(), MultinomialNB(), DecisionTreeClassifier(), SVC()]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
for l in learners:
    pipelines.append(Pipeline([
                          ('vect', CountVectorizer()),
                          # ('vect', HashingVectorizer(non_negative=True)),
                          # ('vect', TfidfVectorizer(use_idf=False, sublinear_tf=True)),
                          # ('vect', TfidfVectorizer(min_df=5,
                          #        max_df = 0.8,
                          #        sublinear_tf=True,
                          #        use_idf=True)),
                          ('selection', SelectKBest()),
                          ('learner', l)]))
    p = {'selection__k' : range(1, 50, 1)}
    if isinstance(l, DecisionTreeClassifier):
        p['learner__max_depth'] = range(1, 5)
        p['learner__min_samples_split'] = range(2, 5)
        p['learner__min_samples_leaf'] = range(2, 5)
    params_grid.append(p)


scores = {}
i = 0
while i < len(pipelines):
    param = params_grid[i]
    p = pipelines[i]

    gs = GridSearchCV(p, param, scoring='accuracy')
    gs.fit(X, y)
    score = gs.best_score_

    scores[p] = score

    ## goes to the  next learner
    i = i + 1

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(scores)




#
#
#
# #variavel que chama a funcao e passa a base com todas as variaveis para aplicar o algoritmo de stemming e stopwords e mostrar
# frasescomstemmingtreinamento = aplicastemmer(df)
#
# #visa separar as palavras de uma frase visando a criação da tabela de frequencias de palavras
# #o parâmetro a ser utilizado deve ser o frasescomstestemming
# def buscapalavras(frases):
#     todaspalavras = []
#     for (palavras, polaridade) in frases:
#         todaspalavras.extend(palavras)
#     return todaspalavras
#
# #criar uma variavel palavra chamamos a funcao e passamos como parametro as frases com stemming e trazer a lista simples com todas as palavras e
# #valores repetidos. Esses valores nao poderao ser repetidos. Tem que fazer funcao para extrair as palavras unicas da base de dados.
# palavrastreinamento = buscapalavras(frasescomstemmingtreinamento)
#
# #devemos extrair as palavras unicas da base de dados e com isso eliminar as repetidas.
# #para isto usamos a função que busca a frequencia das palavras (quantas vezes aparecem na base) e depois pegamos só a palavra
# #usa como parametro a lista de palavras obtida na função anterior
# def buscafrequencia(palavras):
#     #verifica quantas vezes a palavra aparece na base
#     palavras = nltk.FreqDist(palavras)
#     return palavras
#
# frequenciatreinamento = buscafrequencia(palavrastreinamento)
#
# def buscapalavrasunicas(frequencia):
#     # elimina a repetição de palavras - pega só a palavra assim uma palavra que
#     # aparece 4 vezes (am:4) será reduzida para só uma eliminando as repetições
#     freq = frequencia.keys()
#     return freq
#
# palavrasunicastreinamento = buscapalavrasunicas(frequenciatreinamento)
#
#
# #ver que palavras existem em cada frase da base
# def veSeTemPalavra(frase):
#     fra = set(frase)
#     caracteristicas = {}
#     for palavras in palavrasunicastreinamento:
#         caracteristicas['%s' % palavras] = (palavras in fra)
#     return caracteristicas
#
# # caracteristicasfrase = veSeTemPalavra(['pop', 'aguard', 'lady', "abert"])
#
# #estas duas bases é que serão usadas para o processamento, medição de accuracy, etc...
# #ou seja são as que serão submetidas aos algoritmos de aprendizagem de máquina
#
#
# basecompletatreinamento = nltk.classify.apply_features(veSeTemPalavra, frasescomstemmingtreinamento)
#
#
# isklearnDT = DecisionTreeClassifier()
# sklearnDT.fit(basecompletatreinamento)
#
#
#
# nltk.DecisionTreeClassifier.train(basecompletatreinamento)
# classificador = nltk.NaiveBayesClassifier.train(basecompletatreinamento)
# print(" ")
# print("Polaridade: ",classificador.labels())
# print(classificador.show_most_informative_features(20))
# print(classificador.most_informative_features(20))

#accuracy do treino
# if debugTreino: print("#accuracy do treino ")
# if debugTreino: print("Classify (basecompletatreinamento): ",nltk.classify.accuracy(classificador, basecompletatreinamento))
#
# #accuracy do teste
# if debugTeste: print("#accuracy do test ")
# if debugTeste: print("Classify (basecompletateste): ",nltk.classify.accuracy(classificador, basecompletateste))

# arqErro = pathAtual +  "Teste" + complemento + "Erros.csv"
# print (arqErro)
# arqc = open(arqErro,'w')
# pos = 0
# neg = 0
# xis = 0
# for (frase, classe) in baseTeste:
#     i = i + 1
#     testestemming =[]
#     stemmer = nltk.stem.RSLPStemmer()
#     for (palavra) in frase.split():
#         comstem = [p for p in palavra.split()]
#         testestemming.append(str(stemmer.stem(comstem[0])))
#         novo = veSeTemPalavra(testestemming)
#     polaridade = classificador.classify(novo)
#     if polaridade == "p": pos = pos + 1
#     if polaridade == "n": neg = neg + 1
#     if polaridade == "x": xis = xis + 1
#     grava = frase + ";"
#     if classe:
#         grava = grava + classe
#     else:
#         grava = grava + "ERRO"
#     if classe != polaridade:
#         grava = grava + ";" + polaridade
#         arqc.write(grava)
#         arqc.write("\n")
#
# arqc.close()
# tot = pos + neg + xis
# posp = pos/tot * 100
# negp = neg/tot * 100
# xisp = xis/tot * 100
# print("")
# print ("Foram avaliadas ", tot, " frases")
# print ("Frases Positivas = ", str(pos).zfill(2),  " (", round(posp,2), "%)")
# print ("Frases Negativas = ", str(neg).zfill(2),  " (", round(negp,2), "%)")
# print ("Frases Neutras   = ", str(xis).zfill(2),  " (", round(xisp,2), "%)")