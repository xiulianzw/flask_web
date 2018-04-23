import re
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings("ignore")

#加载停用词
stop = pickle.load(open("pkl/stopwords.pkl","rb"))

#删除HTML标记和标点符号，去除停用词
def tokenizer(text):
    #去除HTML标记
    text = re.sub("<[^>]*>","",text)
    #获取所有的表情符
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    #删除标点符号
    text = re.sub("[\W]+"," ",text.lower())+" ".join(emoticons).replace("-","")
    #删除停用词
    tokenized = [word for word in text.split() if word not in stop]
    #提取词干
    porter = PorterStemmer()
    #返回去除停用词之后的单词列表
    return [porter.stem(word) for word in tokenized]
#通过HashingVectorizer获取到评论的特征向量
vect = HashingVectorizer(decode_error="ignore",n_features=2**21,
                         preprocessor=None,tokenizer=tokenizer)