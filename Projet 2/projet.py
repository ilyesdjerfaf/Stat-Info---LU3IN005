import math 
import utils
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt


def getPrior(df):
    n=df.shape[0] #taille de l'echantillon 
    m=0 #moyenne
    for i in range(0,n):
        if(df.at[i,'target']==1):
            m+=1
    m=m/n
    
    s=0 # la racine carré de la variance d'echantillon 
    for i in range(0,n):
        s+=math.pow(df.at[i,'target']-m,2)
    
    s=math.sqrt(s/n)
    ecartType=s #*math.sqrt(n/(n-1))     # ecartType :Estimateur de l'ecart type
    margeErreur=1.96*ecartType/math.sqrt(n) # 1.96 grace au tableau de la loi normale(car n est grand sinon on aurait utilisé la loi de 
                                            #  student) pour un intervalle de confiance de 95%
    d=dict()
    d["estimation"]=m
    d["min5pourcent"]=m-margeErreur
    d["max5pourcent"]=m+margeErreur
    return d

def P2D_l(df,attr):
    val=df[attr].unique()
    
    n=df.shape[0]
    dic_target=dict()
    dic_attr1={val[i]:0 for i in range(0,len(val))}
    dic_attr0={val[i]:0 for i in range(0,len(val))}
    cpt1=0
    cpt0=0
    for i in range(0,n):
        if(df.at[i,'target']==1):
            cpt1+=1
            a=df.at[i,attr]
            dic_attr1[a]+=1
        else:
            cpt0+=1
            a=df.at[i,attr]
            dic_attr0[a]+=1
    for v in val:
        dic_attr1[v]/=cpt1
        dic_attr0[v]/=cpt0
        
    dic_target[1]=dic_attr1
    dic_target[0]=dic_attr0
    return dic_target

def P2D_p(df,attr):
    val=df[attr].unique()
    n=df.shape[0]
    dict_att={x:{1:0, 0:0} for x in val}
    for i in range(0,n):
        a=df.at[i,attr]
        if(df.at[i,'target']==1):
            dict_att[a][1]+=1
        else:
            dict_att[a][0]+=1
            
    for v in val:
        cpt=dict_att[v][1]+dict_att[v][0]
        dict_att[v][1]/=cpt
        dict_att[v][0]/=cpt
        
    return dict_att

def nbParams(df,liste=None):
    if(liste==None):
        liste=df.columns
    n=len(liste)
    nb=1
    for i in range(0,n):
        nb*=len(df[liste[i]].unique())
        
    print(n,' variable(s):',nb*8,' octets ',tailleMemoire(nb*8))
    
def tailleMemoire(nbOctets): # convertir de octet-> kiloOctet-> migaOctet-> gigaOctets
    ko=nbOctets//math.pow(2,10)
    s=""
    if(ko!=0):
        o=nbOctets%math.pow(2,10)
        s="= "+str(ko)+"ko "+str(o)+"o"
        mo=ko//math.pow(2,10)
        if(mo!=0):
            ko=ko%math.pow(2,10)
            s="= "+str(mo)+"mo "+str(ko)+"ko "+str(o)+"o"
            go=mo//math.pow(2,10)
            if(go!=0):
                mo=mo%math.pow(2,10)
                s="= "+str(go)+"go "+str(mo)+"mo "+str(ko)+"ko "+str(o)+"o"
    
    return s

def nbParamsIndep(df,liste=None):
    if(liste==None):
        liste=df.columns
    n=len(liste)
    nb=0
    for i in range(0,n):
        nb+=len(df[liste[i]].unique())
        
    print(n,' variable(s):',nb*8,' octets ',tailleMemoire(nb*8))
        
        
def nbParamsNaiveBayes (data,target,attr=None) :

    if attr == None :
        attr = data.columns
        
    if(len(attr)!=0): 
        valeur=2
        for col in attr:
            if(col!='target'):
                valeur += (len(data[col].unique()))*2
    else:
        valeur=2
    print(len(attr),"Variable(s) : ", valeur*8," octets ",tailleMemoire(valeur*8))
    

class APrioriClassifier(utils.AbstractClassifier):
    
    def __init__(self): #df pour trouver la classe majoritaire
        test=pd.read_csv("test.csv")
        d=getPrior(test)
        if(d["estimation"]>0.5):
            self.classMaj=1
        else:
            self.classMaj=0
        
    
    def estimClass(self, attrs):
        return self.classMaj
        
        
    def statsOnDF(self, df):
       
        
        VN=0
        VP=0
        FN=0
        FP=0
        
        for t in df.itertuples():
            dic=t._asdict()
            if(self.estimClass(dic)==0):
                if(dic["target"]==0):
                    VN+=1
                else:
                    FN+=1
            else:
                if(dic["target"]==1):
                    VP+=1
                else:
                    FP+=1
        
        precision=VP/(VP+FP)
        rappel=VP/(VP+FN)
        
        di=dict()
        di["VP"]=VP
        di["VN"]=VN
        di["FP"]=FP
        di["FN"]=FN
        di["Précision"]=precision
        di["Rappel"]=rappel
        return di
        

        
class ML2DClassifier(APrioriClassifier):
    
    def __init__(self,df,attr):
        self.tableP2DL=P2D_l(df,attr)
        self.attr=attr
        
    def estimClass(self,attrs):
        v=attrs[self.attr]
        if(self.tableP2DL[1][v]<=self.tableP2DL[0][v]):
            return 0
        else:
            return 1
    
        
        
class MAP2DClassifier(APrioriClassifier):
    
    def __init__(self,df,attr):
        self.tableP2DP=P2D_p(df,attr)
        self.attr=attr
        
    def estimClass(self,attrs):
        v=attrs[self.attr]
        if(self.tableP2DP[v][1]<=self.tableP2DP[v][0]):
            return 0
        else:
            return 1
        
        
def drawNaiveBayes(df,attr):
    liste=df.columns
    n=len(liste)
    s=""
    for i in range(0,n-1):
        if(liste[i]!=attr):
            s+=attr+"->"+liste[i]+";"
    if(liste[n-1]!=attr):        
        s+=attr+"->"+liste[n-1]
    
    return utils.drawGraph(s)
        
        
        
class MLNaiveBayesClassifier(APrioriClassifier):
    
    def __init__(self,df):
        self.attrs=df.columns
        self.listeTableP2DL=[]
        
        for col in self.attrs:
            if(col!='target'):
                self.listeTableP2DL.append(P2D_l(df,col))
    
    
    def estimProbas(self,attrs):
        dic=dict()
        a=1
        b=1
        i=0
        for col in self.attrs:
            if(col!='target'): 
                if(attrs[col] in self.listeTableP2DL[i][1] and attrs[col] in self.listeTableP2DL[i][0] ):
                    a*=self.listeTableP2DL[i][1][attrs[col]]
                    b*=self.listeTableP2DL[i][0][attrs[col]]
                    i+=1
                else:
                    a=0
                    b=0
                    i+=1
             
                
        dic[0]=b
        dic[1]=a
        return dic
        
        
    def estimClass(self,attrs):
        dic=self.estimProbas(attrs)
        if(dic[0]>=dic[1]):
            return 0
        else:
            return 1

        
#--------------------------------------------------------------------------------------------------------------------------       
def probaAttribut(df): #calcule la probabilité de chaque attribut de chaque colonne du df(dataFrame)
    valeurs=df.columns
    n=df.shape[0]
    dic={valeurs[i]: {} for i in range(0,len(valeurs))}
    for i in range(0,len(valeurs)):
        vals=df[valeurs[i]].unique()
        dic[valeurs[i]]={vals[j]: 0 for j in range(0,len(vals))}
        
    for t in df.itertuples():
        dic_t=t._asdict()
        for i in range(0,len(valeurs)):
            v=dic_t[valeurs[i]]
            dic[valeurs[i]][v]+=1/n
            
    return dic
        
#----------------------------------------------------------------------------------------------------------------------------     
class MAPNaiveBayesClassifier(APrioriClassifier):
    
    def __init__(self,df):
        self.attrs=df.columns
        self.listeTableP2DL=[]
        self.df=df
        self.proba=getPrior(self.df)['estimation']
        self.dfProba=probaAttribut(df)
        for col in self.attrs:
            if(col!='target'):
                self.listeTableP2DL.append(P2D_l(df,col))
                                           
    def estimProbas(self,attrs):
        dic=dict()
        a=1
        b=1
        i=0
        for col in self.attrs:
            if(col!='target'):
                if(attrs[col] in self.listeTableP2DL[i][0] and attrs[col] in self.listeTableP2DL[i][1]):
                    a*=self.listeTableP2DL[i][1][attrs[col]]/self.dfProba[col][attrs[col]]
                    b*=self.listeTableP2DL[i][0][attrs[col]]/self.dfProba[col][attrs[col]]
                    i+=1
                else:
                    a=0
                    b=0
                    i+=1
        
        proba=self.proba             
        dic[0]=b*(1-proba)
        dic[1]=a*proba
        return dic

    def estimClass(self,attrs):
        dic=self.estimProbas(attrs)
        if(dic[0]>=dic[1]):
            return 0
        else:
            return 1
#--------------------------------------------------------------------------------------------------------------------
def isIndepFromTarget(df,attr,x):
    """
    Verifie si attr est indépendant de target au seuil de x%    
    df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
    attr: le nom d'une colonne du dataframe df.
    x: seuil de confiance.
    """
    listeVal=df[attr].unique()
    matContengence=np.zeros((2,listeVal.size), dtype=int)
    dictVal={listeVal[i]: i for i in range(len(listeVal))}
    
    for row in df.itertuples():
        rowDic=row._asdict()
        matContengence[rowDic['target'],dictVal[rowDic[attr]]]+=1
        
    _,p,_,_= scipy.stats.chi2_contingency(matContengence)
    
    return p>x
    
#---------------------------------------------------------------------------------------------------
class ReducedMLNaiveBayesClassifier(APrioriClassifier):
    
    def __init__(self,df,x):
        self.attrs=df.columns
        self.listeTableP2DL=[]
        self.df=df
        self.df_=df.copy(deep=True)
        y=False
        
        for col in self.attrs:
            if(col!='target'):
                y=isIndepFromTarget(df,col,x)
                if(not y):
                    self.listeTableP2DL.append(P2D_l(df,col))
                else:
                    self.df_.drop([col],1,inplace=True)
                
        self.attrs=self.df_.columns
       
        
        
    def estimProbas(self,attrs):
        dic=dict()
        a=1
        b=1
        i=0
        for col in self.attrs:
            if(col!='target'): 
                if(attrs[col] in self.listeTableP2DL[i][1] and attrs[col] in self.listeTableP2DL[i][0] ):
                    a*=self.listeTableP2DL[i][1][attrs[col]]
                    b*=self.listeTableP2DL[i][0][attrs[col]]
                    i+=1
                else:
                    a=0
                    b=0
                    i+=1
             
                
        dic[0]=b
        dic[1]=a
        return dic
        
        
    def estimClass(self,attrs):
        dic=self.estimProbas(attrs)
        if(dic[0]>=dic[1]):
            return 0
        else:
            return 1

    def draw(self):
        return drawNaiveBayes(self.df_,'target')
        
#----------------------------------------------------------------------------------------------------------

class ReducedMAPNaiveBayesClassifier(APrioriClassifier):
    
    def __init__(self,df,x):
        self.attrs=df.columns
        self.listeTableP2DL=[]
        self.df=df
        self.df_=df.copy(deep=True)
        
        self.proba=getPrior(self.df)['estimation']
        self.dfProba=probaAttribut(df)
        
        
        y=False
        
        for col in self.attrs:
            if(col!='target'):
                y=isIndepFromTarget(df,col,x)
                if(not y):
                    self.listeTableP2DL.append(P2D_l(df,col))
                    
                else:
                    self.df_.drop([col],1,inplace=True)
                     
        self.attrs=self.df_.columns
        
    def estimProbas(self,attrs):
        dic=dict()
        a=1
        b=1
        i=0
        for col in self.attrs:
            if(col!='target'):
                if(attrs[col] in self.listeTableP2DL[i][0] and attrs[col] in self.listeTableP2DL[i][1]):
                    a*=self.listeTableP2DL[i][1][attrs[col]]/self.dfProba[col][attrs[col]]
                    b*=self.listeTableP2DL[i][0][attrs[col]]/self.dfProba[col][attrs[col]]
                    i+=1
                else:
                    a=0
                    b=0
                    i+=1
        
                     
        dic[0]=b*(1-self.proba)
        dic[1]=a*self.proba
        return dic

    def estimClass(self,attrs):
        dic=self.estimProbas(attrs)
        if(dic[0]>=dic[1]):
            return 0
        else:
            return 1
        
        
    def draw(self):
        return drawNaiveBayes(self.df_,'target')    
        
        
#---------------------------------------------------------------------------------------

def mapClassifiers(dic,df):
    precision=[]
    rappel=[]
    
    for i in dic:
        precision.append(dic[i].statsOnDF(df)['Précision'])
        rappel.append(dic[i].statsOnDF(df)['Rappel'])
        
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Précision")
    ax.set_ylabel("Rappel")
    ax.scatter(precision, rappel, marker = 'x', c = 'blue') 
    for i, nom in enumerate(dic):
        ax.annotate(nom, (precision[i], rappel[i]))
    
    plt.show()
        
        
#----------------------------------------------------------------------------------------

def MutualInformation(df,x,y):
    n=df.shape[0] #nombre de lignes du df: dataFrame
    information=0.0
  
    
    for i in df[x].unique():
        px=0
        pyy=0
        pxy=0
        for j in df[y].unique():
            px=len(df[df[x]==i])/n
            py=len(df[df[y]==j])/n
            pxy=len(df[df[x]==i][df[y]==j])/n
            if(px!=0 and py!=0 and pxy!=0):
                information+=pxy*(math.log2(pxy)-math.log2(px*py))
    return information
        
        
        
        
#------------------------------------------------------------------------------------------

def ConditionalMutualInformation(df,x,y,z):
    n=df.shape[0]
    information=0
    
    for i in df[x].unique():
        pxz=0
        pyz=0
        pz=0
        pxyz=0
        
        for j in df[y].unique():
            for k in df[z].unique():
                pxz=len(df[df[x]==i][df[z]==k])/n
                pyz=len(df[df[y]==j][df[z]==k])/n
                pz=len(df[df[z]==k])/n
                pxyz=len(df[df[x]==i][df[y]==j][df[z]==k])/n
                if(pxz!=0 and pyz!=0 and pz!=0 and pxyz!=0):
                    information+=pxyz*math.log2(pz*pxyz/(pxz*pyz))
                    
    return information
    
    
    
#-----------------------------------------------------------------------------------

def MeanForSymetricWeights(matrice):
    
    mean=0
    for i in range(0,len(matrice)):
        for j in range(0,len(matrice[0])):
            mean+=matrice[i][j]
    mean/=(len(matrice)*len(matrice[0])-len(matrice))  #on divise pas le nombre d'éléments - la diagonale
    return mean

#----------------------------------------------------------------------------------

def SimplifyConditionalMutualInformationMatrix(matrice):
    mean=MeanForSymetricWeights(matrice)
    
    for i in range(len(matrice)):
        for j in range(len(matrice[0])):
            if(matrice[i][j]<mean):
                matrice[i][j]=0
                
#-----------------------------------------------------------------------------------
    
def Kruskal(df,matrice):
    SimplifyConditionalMutualInformationMatrix(matrice)
    graph = [] #le graph
    listKruskal = [] #la liste des arcs
    groupes = {}
    liste = df.columns
    for i in range(0,len(matrice)):
        for j in range(0,len(matrice)):
            if(matrice[j][i]!=0):
                graph.append([j,i,matrice[i][j]])
    graph = list(sorted(graph,key=takeThirdElement,reverse=True)) # on ordonne la liste du plus grand poids au plus petit 
    
    for x,y,poid in graph:
        if(not find(x,y,groupes)): # si pas de circuit formé
            listKruskal.append((liste[x],liste[y],poid))
            union(x,y,groupes)
    return listKruskal

#---------------------------------------------------------------------------------
def takeThirdElement(elem):# utiliser pour ordonner la liste des arcs dans le graphe( fonction: Kruskal) selon le poids de l'arc
    return elem[2]
#-----------------------------------------------------------------------

def union(x,y,groupes):

    if not x in groupes and not y in groupes:
        groupes[x] =y
        groupes[y] =y
    elif not x in groupes :
        if not y in groupes :
            groupes[x]=y
            groupes[y]=y
        else:
            groupes[x]=groupes[y]
    elif not y in groupes:
        groupes[y]=groupes[x]
    else:
        for key in groupes :
            if groupes[key]==groupes[x]:
                groupes[key] = groupes[y]
    
#-------------------------------------------------------------------------

def find (x,y,groupes):  #retourne True si x et y sont dans le meme groupe 
    if not x in groupes or not y in groupes :
        return False
    return groupes[x]==groupes[y]
    
#--------------------------------------------------------------------------
    
        