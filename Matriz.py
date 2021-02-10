import numpy as np
import pandas as pd

class matriz:

    def __init__(self, time, n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,
                 n26,n27,n28,amount):
        self.time = time
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4
        self.n5 = n5
        self.n6 = n6
        self.n7 = n7
        self.n8 = n8
        self.n9 = n9
        self.n10 = n10
        self.n11 = n11
        self.n12 = n12
        self.n13 = n13
        self.n14 = n14
        self.n15 = n15
        self.n16 = n16
        self.n17 = n17
        self.n18 = n18
        self.n19 = n19
        self.n20 = n20
        self.n21 = n21
        self.n22 = n22
        self.n23 = n23
        self.n24 = n24
        self.n25 = n25
        self.n26 = n26
        self.n27 = n27
        self.n28 = n28
        self.amount = amount

    #retorna os dados para as camadas de entrada
    def lines_enter(self):
        return np.array([self.time, self.n1, self.n2, self.n3, self.n4, self.n5, self.n6, self.n7, self.n8, self.n9, self.n10,
                         self.n11, self.n12, self.n13, self.n14, self.n15, self.n16, self.n17, self.n18, self.n19, self.n20,
                         self.n21, self.n22, self.n23, self.n24, self.n25, self.n26, self.n27, self.n28, self.amount]).reshape(30,1)

    #retorna os pesos para camada de entrada
    def matriz_pEnter(self):
        pEnter = pd.read_csv("Datasets/pesosOculta.csv")
        return pEnter

    #retorna os pesos da camada oculta
    def matriz_pOcult(self):
        pOcult = pd.read_csv("Datasets/pesosSaida.csv")
        return pOcult

    #retorna o bias da camada de entrada
    def biasEnter(self):
        biasEnter = pd.read_csv("Datasets/biasOculta.csv")
        return biasEnter

    #retorna o bias da camada oculta
    def biasOcult(self):
        biasOcult = pd.read_csv("Datasets/biasSaida.csv")
        return biasOcult

    #retorna a multiplicação de uma matriz AxB
    def mult(self, A, B):
        return np.dot(A,B)

    #retorna a soma de uma matriz AxB
    def sum(self,A, B):
        return np.add(A,B)

    #reotnra a subtração de uma matriz AxB
    def substract(self, A, B):
        return np.subtract(A,B)

    #retorna a sigmoid de X
    def sigmoid(self, X):
        teste = 1/(1+ np.exp(-X))
        return teste

    #retorna a sigmoid de X
    def dSigmoid(self , X):
        return X * (1-X)

    #retorna o hadamard de AxB
    def hadamard(self, A, B):
        return A * B

    #retonra o escalar de AxB
    def escalar_multi(self, A, x):
        return A * x

    #retonrna uma matriz transposta
    def transpose(self, A):
        return  np.transpose(A)
