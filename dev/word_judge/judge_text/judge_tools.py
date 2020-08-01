import pandas as pd
# from sklearn.svm import SVR
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np
from janome.tokenizer import Tokenizer
import warnings
import sys
warnings.simplefilter('ignore')
# sys.path.append('/Users/nagataeiki/.pyenv/versions/3.7.4/lib/python3.7/site-packages')
from tqdm import tqdm 
import gensim
import pickle
from gensim.models import word2vec
from janome.tokenizer import Tokenizer
tokenizer=Tokenizer()
b_l = pd.read_csv('./judge_text/b_l.csv').T.values[0]
model = word2vec.Word2Vec.load("./judge_text/models/100000.model")


def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def mophologicize(text):
    words = []
    for t in tokenizer.tokenize(text):
        words.append(t.surface)    
    return words



def create_mean_vector_data(df):
    col=[]
    
    for i in range(100):
        col.append("f_"+str(i))

    sandbox=pd.DataFrame(columns=col)
    for r in tqdm(df.iterrows()):
        text = r[1].mophologic_content
        vec_list=[]
        for t in text:
                try:
                    vec = model.wv[t]
                    vec_list.append(vec)
                except:
                    vec_list.append(np.zeros(100))
        vectors = pd.DataFrame(np.array(vec_list), columns=col)
        tmp = pd.DataFrame(pd.DataFrame(vectors, columns=col).mean(axis=0)).T
        sandbox = pd.concat([sandbox, tmp])
        

    sandbox.insert(0, "target",df.target.values)
    sandbox.insert(1,"content",df.content.values)
    sandbox["id"]= df.id.values
    return sandbox


def create_mean_cosine_sim_data(df):
    base = pd.DataFrame(columns=np.array(b_l))
    for r in tqdm(df.iterrows()):
        text = r[1].mophologic_content
        ave_list=[]

        for word in b_l:
            try:
                v1 = model.wv[word]
                cos_list = []
                for t in text:
                    try:
                        v2 = model.wv[t]
                        cos_list.append(cos_similarity(v1,v2))
                    except:
                        pass
            except:
                pass
            ave_list.append(np.array(cos_list).mean())
        tmp=pd.DataFrame(np.array(ave_list).reshape(1,-1), columns=b_l)
        base=pd.concat([base, tmp])
        
    base["target"] =df.target.values
    base["content"] = df.content.values
    base = base.loc[:,base.columns[::-1]]
    base["id"] = df.id.values
    return base


def create_the_highest_cosine_sim_data(df):
    base = pd.DataFrame(columns=np.array(b_l))
    for r in tqdm(df.iterrows()):
        text = r[1].mophologic_content
        highest_list=[]

        for word in b_l:
            try:
                v1 = model.wv[word]
                cos_list = []
                for t in text:
                    try:
                        v2=model.wv[t]
                        cos_list.append(cos_similarity(v1,v2))
                    except:
                        pass
            except:
                pass
            try:
                highest_list.append(np.array(cos_list).max())
            except:
                highest_list.append(np.nan)
                
        tmp=pd.DataFrame(np.array(highest_list).reshape(1,-1), columns=b_l)
        base=pd.concat([base, tmp])
        
    base["target"] =df.target.values
    base["content"] = df.content.values
    base = base.loc[:,base.columns[::-1]]
    base["id"] = df.id.values
    return base

def predict_new_data(text):
    mophologics = mophologicize(text)
    col=["content", "mophologic_content"]
    text_df=pd.DataFrame(np.array([text, mophologics]), ).T

    text_df.columns = col
    text_df["target"]=np.nan
    text_df["id"]=np.nan
    fe1=create_mean_vector_data(text_df).drop(['id',"content","target"], axis=1)
    fe2=create_the_highest_cosine_sim_data(text_df).drop(['id',"content","target"], axis=1)
    fe3=create_mean_cosine_sim_data(text_df).drop(['id',"content","target"], axis=1)

    m1 = pickle.load(open('./judge_text/models/svm_1.sav',"rb"))
    m2 = pickle.load(open('./judge_text/models/svm_2.sav',"rb"))
    m3 = pickle.load(open('./judge_text/models/svm_3.sav',"rb"))
    p_1=m1.predict(fe1)
    p_2=m2.predict(fe2)
    p_3=m3.predict(fe3)

    return (p_1+p_2+p_3)/3
    


def judgefunc(text):
    if len(text)<1:
        return "文字数が少なすぎます"
    else:
        return predict_new_data(text)