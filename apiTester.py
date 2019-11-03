# importing the requests library
import requests

# api-endpoint
URL = "http://127.0.0.1:5000/getClassificacao"


colunas = ['C1','N1','C2','C3','N2','C4','C5','N3','C6','C7','N4','C8','N5','C9','C10','N6','C11','N7','C12','C13','LABEL']


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import csv




def leDataSet():
    df = pd.read_csv('DatasetML.csv')
    df.columns = colunas
    # no dataset original tinhas dois c5
    return df

dfteste=  leDataSet()
erros =0
total =0
classe1 =0
classe2 =0
numerotestes = 50
indices = np.random.randint(1000, size=numerotestes)

for i in indices:
    PARAMS= dfteste.iloc[i].to_dict()
    true_cla = PARAMS['LABEL']
    r = requests.get(url=URL, params=PARAMS)
    data = r.json()

    p = int(data['resultado'])
    if  p !=  true_cla:
        erros = erros +1
    if true_cla ==1 :
        classe1 = classe1+ 1
    elif true_cla ==2 :
        classe2 = classe2 + 1


print(str(classe1) + ' testes com classe 1 '  )
print(str(classe2) + ' testes com classe 2 '  )

print(str(numerotestes -erros) + ' acertos de  ' + str(numerotestes) )









