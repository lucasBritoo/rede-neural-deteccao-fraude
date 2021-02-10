import matplotlib.pyplot as plt

#etapa de fuzyficação
temperatura = {'frio':[0,15,40],'media':[15,30,55],'quente':[25,45,60]}

temperatura['frio']

indices=list(temperatura.keys())
valores=list(temperatura.values())


fig, ax = plt.subplots(figsize=(10,5))


ax.plot(valores[0],[0,1,0],c='g',label=indices[0])
ax.plot(valores[1],[0,1,0],c='y',label=indices[1])
ax.plot(valores[2],[0,1,0],c='r',label=indices[2])

ax.set_title('Funçções de pertinência')
ax.set_ylabel('pertinência')
ax.set_xlabel('temperatura')
ax.legend(['frio','media','quente'])
