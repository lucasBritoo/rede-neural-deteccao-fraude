from Matriz import matriz
import pandas as pd

class neuralNetwork:

    def __init__(self, df, out, learning_rate):
        self.df = df  # objeto do dataset de treinamento
        self.out = out  # objeto do dataset de resposta
        self.learning_rate = learning_rate  # parâmetro de aprendizado


    #inserção dos dados
    def enterData(self, nDataFrame, index, geration):

        #define quantidade de invididuos testados
        while index < nDataFrame:

            #extraçãdo dos dados do dataset
            time = self.df.at[index, 'Time']
            v1 = self.df.at[index, 'V1']
            v2 = self.df.at[index, 'V2']
            v3 = self.df.at[index, 'V3']
            v4= self.df.at[index, 'V4']
            v5= self.df.at[index, 'V5']
            v6= self.df.at[index, 'V6']
            v7= self.df.at[index, 'V7']
            v8= self.df.at[index, 'V8']
            v9= self.df.at[index, 'V9']
            v10= self.df.at[index, 'V10']
            v11= self.df.at[index, 'V11']
            v12= self.df.at[index, 'V12']
            v13= self.df.at[index, 'V13']
            v14= self.df.at[index, 'V14']
            v15= self.df.at[index, 'V15']
            v16= self.df.at[index, 'V16']
            v17= self.df.at[index, 'V17']
            v18= self.df.at[index, 'V18']
            v19= self.df.at[index, 'V19']
            v20= self.df.at[index, 'V20']
            v21= self.df.at[index, 'V21']
            v22= self.df.at[index, 'V22']
            v23= self.df.at[index, 'V23']
            v24= self.df.at[index, 'V24']
            v25= self.df.at[index, 'V25']
            v26= self.df.at[index, 'V26']
            v27= self.df.at[index, 'V27']
            v28= self.df.at[index, 'V28']
            amount = self.df.at[index, 'Amount']

            mtz = matriz(time, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,
                         v25,v26,v27,v28, amount)  # define objeto do individuo

            #leitura da saida correta e do dataset de entrada
            saidaCorreta = self.out.at[index, 'Class']                       #extração da resposta correta para o individuo

            #leitura dos pesos de entrada e bias
            pOculta = mtz.matriz_pEnter()  # incializa matriz dos pesos da (entrada -> oculta)
            pSaida = mtz.matriz_pOcult()  # inicializa matriz dos pesos da (oculta -> saida)
            biasOculta = mtz.biasEnter()  # le  os valores definidos no bias (entrada -> oculta)
            biasSaida = mtz.biasOcult()  # le os valores definidos no bias (oculta -> saida)

            #incializa processo de feedForward
            self.treino(mtz, saidaCorreta, pOculta, pSaida, biasOculta, biasSaida, index, geration)
            index+=1                #contador de indivíduos


    def treino(self,mtz, saidaCorreta, pOculta, pSaida, biasOculta, biasSaida, index, geration):

        #FeedForward
        entrada = mtz.lines_enter()                                     # inicializa matriz de entrada

        oculta = mtz.mult(pOculta, entrada)
        oculta = mtz.sum(oculta, biasOculta)
        oculta = mtz.sigmoid(oculta)

        saida = mtz.mult(pSaida, oculta)
        saida = mtz.sum(saida, biasSaida)
        saida = mtz.sigmoid(saida)
        saida = saida.iat[0,0]


        #BackPropagation

        #saida -> oculta
        erro_saida = saidaCorreta - saida
        d_saida = mtz.dSigmoid(saida)

        oculta_T = mtz.transpose(oculta)
        gradiente = mtz.hadamard(erro_saida, d_saida)
        gradiente = gradiente * self.learning_rate
        biasSaida = mtz.sum(biasSaida, gradiente)
        pesos_s_deltas = mtz.mult(gradiente, oculta_T)
        pSaida = mtz.sum(pSaida, pesos_s_deltas)



        #oculta -> entrada
        pesos_o_T = mtz.transpose(pSaida)
        erro_oculta = mtz.mult(pesos_o_T, erro_saida)

        d_oculta = mtz.dSigmoid(oculta)
        entrada_T = mtz.transpose(entrada)
        gradiente_oculta = mtz.hadamard(erro_oculta, d_oculta)
        gradiente_oculta = gradiente_oculta * self.learning_rate
        biasOculta = mtz.sum(biasOculta, gradiente_oculta)

        pesos_o_deltas = mtz.mult(gradiente_oculta, entrada_T)
        pOculta = mtz.sum(pOculta, pesos_o_deltas)

        self.save(pOculta, pSaida, biasOculta, biasSaida)

        print('{} - Index: {}   SC: {}   SG: {}'.format(geration, index, saidaCorreta, saida))


    def save(self, pOculta, pSaida, biasOculta, biasSaida):
        pes = pd.DataFrame(pOculta)
        pes.to_csv("Datasets/pesosOculta.csv", index=False)

        pes1 = pd.DataFrame(pSaida)
        pes1.to_csv("Datasets/pesosSaida.csv", index=False)

        pes2 = pd.DataFrame(biasOculta)
        pes2.to_csv("Datasets/biasOculta.csv", index=False)

        pes3 = pd.DataFrame(biasSaida)
        pes3.to_csv("Datasets/biasSaida.csv", index=False)


