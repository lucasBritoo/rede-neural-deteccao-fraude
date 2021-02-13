import pandas as pd
from NeuralNetwork import neuralNetwork

df = pd.read_csv("Datasets/dataset_treino.csv")                #leitura do dataset final
out = pd.read_csv("Datasets/dataset_respostaTreino.csv")       #leitura do dataset de resposta final

#definição dos parametros para treinamento da rede
nDataFrame = df.shape[0]                                #retorna quantidade de elementos do dataset
maxGeneration = 10000                                       #define o limite de gerações
xGeneration = 0                                         #contador de gerações
minError = 0.25                                         #define a taxa de erro permitida em cada teste no individuo
learning_rate = 0.8                                       #taxa de aprendizado
nNet = neuralNetwork(df, out, learning_rate)  #define construtor para rede neural

#laço principal
while xGeneration < maxGeneration:                      #while para contagem de gerações
    nNet.enterData(nDataFrame, 0 , xGeneration)        #inicializa o processo de FeedForward para o individuo
    nNet.contador2 = 0;
    nNet.contador = 0;

    xGeneration+=1                                      #contador de gerações
