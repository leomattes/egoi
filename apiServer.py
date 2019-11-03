from flask import Flask
from flask import request
from flask import json
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


app = Flask(__name__)


oneHotEncoders = None
classificador= None
msgerr =  ''

def carrega():


    file = open('OneHotCodificadores', 'rb')
    oneHotEncoders = pickle.load(file)
    file.close()
    infile = open('ClassificadorEgoi', 'rb')
    classificador = pickle.load(infile)
    infile.close()
    return oneHotEncoders, classificador
    #except:
     #   msgerr = 'eror ao carregar arquivo'

colunasN = ['N1','N2','N3','N4','N5','N6','N7']
colunasC = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13']



def parametersToDic( args):
    parametros = {}
    for c in colunasN:
        if c in args:
            v =  str(args[c])
            if v.isnumeric() :
                parametros[c] = int(v)
            else:
                return None, 'Parametro inválido ' + c
        else:
            return None, 'Parametro inválido ' + c

    for c in colunasC:
        if c in args:
           parametros[c] = args[c]
        else:
            return None, 'Parametro inválido ' + c

    return parametros, ''


def getNumericValues(dic):
    vetc = np.zeros(0)
    for c in colunasN:
        valor = int(dic[c])
        vetc = np.concatenate((vetc, np.array([valor])), axis=0)
    return vetc


def getoneHotColum(nome, valor ):
    enc = oneHotEncoders[nome]
    col_enc =  enc.transform([[valor]])
    return col_enc.toarray()


def getoneHotVector(dic, vetc):
    for c in colunasC:
        valor = dic[c]
        array = getoneHotColum(c, valor)
        vetc = np.concatenate((vetc, array.flatten()), axis=0)
    return vetc


def encodeDic(dic):
    vetc = getNumericValues(dic)
    vetc = getoneHotVector(dic, vetc)
    vetc = vetc.reshape((1, 61))
    return vetc


def getResposta(msg, resultado):
    resposta = {
        "msg": msg,
        "resultado": resultado
    }

    response = app.response_class(
        response=json.dumps(resposta),
        status=200,
        mimetype='application/json'
    )

    return response

@app.route('/getClassificacao', methods=['GET', 'POST'])
def getClassificacao():
    global classificador
    global oneHotEncoders
    if classificador ==  None:
        oneHotEncoders, classificador = carrega();
    if classificador == None:
        return getResposta( 'Erro ao carergar classificador : ' + msgerr, -1)


    parametros, msg = parametersToDic( request.args)
    if  parametros != None :
        X = encodeDic(parametros)
        r = classificador.predict(X)
        return getResposta( 'Label predita  : ' , str(r[0]))


    else:
        return getResposta( 'Eror de parametrização : ' + msg, -1)










if __name__ == "__main__":

    app.run()

