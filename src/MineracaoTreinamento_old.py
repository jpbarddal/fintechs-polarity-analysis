# coding: utf-8
import nltk
import csv
import time
from datetime import datetime
#nltk.download()

inicio = time.time()

debug = True
debugTreino = True
debugTeste = True
debugQtd = 5
if debug: print (inicio)
execAvaliacao = False

#pathAtual = "E:/Python/Marina/ArquivoTCC/BasesProjetoFinal/"
# pathAtual = "C:/Users/mpseara/PycharmProjects/MineracaoTCC/"
pathAtual = "./"
unificada ="Modelo"
neon = "_Neon"
next = "_Next"
original = "_Original"

#"================================== Aqui troca arquivo a ser processado <<<<<<<<<<<<<<<<<<<<<
complemento = unificada

arqTreinoAtual = "Treino" + complemento + ".csv"
pathTreino = pathAtual + arqTreinoAtual
arqTesteAtual = "Teste" + complemento + ".csv"
pathTeste = pathAtual + arqTesteAtual

print ("==============================================")
print ("===", datetime.now())
print ("=== Usando arquivo Treino: ", pathTreino)
print ("=== Usando arquivo Teste : ", pathTeste)
print ("==============================================")

#================================================
#base de dados utilizada para treinar o algoritmo
#================================================
baseTreino = []
ArqTreino = []
with open(pathTreino,'r') as simple:
    ArqTreino = csv.reader(simple)
    for row in ArqTreino:
       tblb = ()
       tblb = (row[0],row[1])
       baseTreino.append(tblb)
if debugTreino: print ("1ª linha baseTreino: ",baseTreino[0])

#================================================
#base de dados utilizada para testar o algoritmo
#================================================

baseTeste = []
ArqTeste = []
with open(pathTeste,'r') as simple:
    ArqTeste = csv.reader(simple)
    for row in ArqTeste:
   # print 'row1 ', row

       tblb = (row[0],row[1])
       baseTeste.append(tblb)
if debugTeste: print ("95ª linha baseTeste: ",baseTeste[95])

#remocao das stopswords
#uso de stopwords do nltk - ja possui recursos para trabalhar com acentuacao em portugues
stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
if debug:
    print("1.===== stopwordsnltk (portuguese) =====")
    print (stopwordsnltk)
#Ajusta stopwords acrescentando novas palavras identificadas
ajustaStopWords = pathAtual + 'minhasStopWords.txt'
with open(ajustaStopWords,'r') as simple:
    ArqStopWords = csv.reader(simple)
    for registro in ArqStopWords:
       stopwordsnltk.append(registro[0])
if debug: print(stopwordsnltk)
#Remove stopWords ==> para ver como funciona a extração mesmo vai ocorrer na função aplicastemmer
p1 = 0
def removestopwords(naFrase):
    global p1
    frases = []
    for (palavras, emocao) in naFrase:
        print palavras
        p1 = p1 + 1
        #split serve para quebrar as palavras, ele acha um espaco em branco e quebra - passar o stopwordsnltk para usar a lista do nltk.
        #print(emocao)
        if debug:
            if p1 < debugQtd:
                print ("1.1===== palavras + polaridade =====", p1, palavras, emocao)
                #print (palavras, emocao)
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        if debug:
            if p1 < debugQtd:
                print ("1.2===== semstop ===================", p1, semstop)
                #print (semstop)
        #adicionar dentro da variavel frases - passa como parametro emocao para manter a classe
        frases.append((semstop, emocao))
        if debug:
            if p1 < debugQtd:
                print("1.3 ===== frases sem stopwords =====",p1, frases)
                #print (frases)
    return frases
#chamando a funcao stopwords
if debugTreino: print("2.Remove stopwords Treino: ", removestopwords(baseTreino))
if debugTeste: print("2.Remove stopwords Teste: ", removestopwords(baseTeste))

#aplica stemming
def aplicastemmer(texto):
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
#variavel que chama a funcao e passa a base com todas as variaveis para aplicar o algoritmo de stemming e stopwords e mostrar
frasescomstemmingtreinamento = aplicastemmer(baseTreino)
frasescomstemmingteste = aplicastemmer(baseTeste)
if debugTreino: print("3.Frases de treinamento após stemming : ", frasescomstemmingtreinamento)
if debugTeste:  print("3.Frases de teste após stemming ..... : ", frasescomstemmingteste)

#visa separar as palavras de uma frase visando a criação da tabela de frequencias de palavras
#o parâmetro a ser utilizado deve ser o frasescomstestemming
def buscapalavras(frases):
    todaspalavras = []
    for (palavras, polaridade) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras
#criar uma variavel palavra chamamos a funcao e passamos como parametro as frases com stemming e trazer a lista simples com todas as palavras e
#valores repetidos. Esses valores nao poderao ser repetidos. Tem que fazer funcao para extrair as palavras unicas da base de dados.
palavrastreinamento = buscapalavras(frasescomstemmingtreinamento)
palavrasteste = buscapalavras(frasescomstemmingteste)
if debugTreino: print("4.Palavras treinamento: " ,palavrastreinamento)
if debugTeste:  print("4.Palavras teste: ..... " ,palavrasteste)

#devemos extrair as palavras unicas da base de dados e com isso eliminar as repetidas.
#para isto usamos a função que busca a frequencia das palavras (quantas vezes aparecem na base) e depois pegamos só a palavra
#usa como parametro a lista de palavras obtida na função anterior
def buscafrequencia(palavras):
    #verifica quantas vezes a palavra aparece na base
    palavras = nltk.FreqDist(palavras)
    return palavras

frequenciatreinamento = buscafrequencia(palavrastreinamento)
frequenciateste = buscafrequencia(palavrasteste)

if debugTreino: print("5.Frequencia de palavras treinamento: ", frequenciatreinamento.most_common(debugQtd))
if debugTeste:  print("5.Frequencia de palavras teste: ..... ", frequenciateste.most_common(debugQtd))

def buscapalavrasunicas(frequencia):
    #elimina a repetição de palavras - pega só a palavra assim uma palavra que aparece 4 vezes (am:4) será reduzida para só uma eliminando as repetições
    freq = frequencia.keys()
    return freq

palavrasunicastreinamento = buscapalavrasunicas(frequenciatreinamento)
palavrasunicasteste = buscapalavrasunicas(frequenciateste)
if debugTeste:  print("6.Palavras únicas Teste : ",palavrasunicasteste)
if debugTreino: print("6.Palavras únicas Treino: ",palavrasunicastreinamento)
#ver que palavras existem em cada frase da base
def veSeTemPalavra(frase):
    fra = set(frase)
    caracteristicas = {}
    for palavras in palavrasunicastreinamento:
        caracteristicas['%s' % palavras] = (palavras in fra)
    return caracteristicas

caracteristicasfrase = veSeTemPalavra(['pop', 'aguard', 'lady', "abert"])
if debugTeste or debugTreino: print(caracteristicasfrase)

#estas duas bases é que serão usadas para o processamento, medição de accuracy, etc...
#ou seja são as que serão submetidas aos algoritmos de aprendizagem de máquina
basecompletatreinamento = nltk.classify.apply_features(veSeTemPalavra, frasescomstemmingtreinamento)
basecompletateste = nltk.classify.apply_features(veSeTemPalavra, frasescomstemmingteste)
cf = 10
i=0
if debugTeste:
    print("===== Tabela características base Teste =====")
    for i in range(cf):
        print(basecompletateste[i])
if debugTreino:
    print("===== Tabela características base Treino =====")
    for i in range(cf):
        print(basecompletatreinamento[i])

# constroi a tabela de probabilidade
classificador = nltk.NaiveBayesClassifier.train(basecompletatreinamento)
if debugTreino: print(" ")
if debugTreino: print("Polaridade: ",classificador.labels())
if debugTreino: print(classificador.show_most_informative_features(debugQtd*2))
if debugTreino: print(classificador.most_informative_features(debugQtd*2))

#accuracy do treino
if debugTreino: print("#accuracy do treino ")
if debugTreino: print("Classify (basecompletatreinamento): ",nltk.classify.accuracy(classificador, basecompletatreinamento))

#accuracy do teste
if debugTeste: print("#accuracy do test ")
if debugTeste: print("Classify (basecompletateste): ",nltk.classify.accuracy(classificador, basecompletateste))

arqErro = pathAtual +  "Teste" + complemento + "Erros.csv"
print (arqErro)
arqc = open(arqErro,'w')
pos = 0
neg = 0
xis = 0
for (frase, classe) in baseTeste:
    i = i + 1
    testestemming =[]
    stemmer = nltk.stem.RSLPStemmer()
    for (palavra) in frase.split():
        comstem = [p for p in palavra.split()]
        testestemming.append(str(stemmer.stem(comstem[0])))
        novo = veSeTemPalavra(testestemming)
    polaridade = classificador.classify(novo)
    if polaridade == "p": pos = pos + 1
    if polaridade == "n": neg = neg + 1
    if polaridade == "x": xis = xis + 1
    grava = frase + ";"
    if classe:
        grava = grava + classe
    else:
        grava = grava + "ERRO"
    if classe != polaridade:
        grava = grava + ";" + polaridade
        arqc.write(grava)
        arqc.write("\n")

arqc.close()
tot = pos + neg + xis
posp = pos/tot * 100
negp = neg/tot * 100
xisp = xis/tot * 100
print("")
print ("Foram avaliadas ", tot, " frases")
print ("Frases Positivas = ", str(pos).zfill(2),  " (", round(posp,2), "%)")
print ("Frases Negativas = ", str(neg).zfill(2),  " (", round(negp,2), "%)")
print ("Frases Neutras   = ", str(xis).zfill(2),  " (", round(xisp,2), "%)")










fim = time.time()
if debug: print(fim)
duracao = fim - inicio
min = int(duracao / 60)
seg = int(duracao - min*60)
#print (fim-inicio)
print("")
print ("=======================================")
print ("==> Tempo: ", min , "minutos e ", seg , "segundos ===")
print ("=======================================")