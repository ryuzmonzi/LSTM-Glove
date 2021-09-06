# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:34:43 2021

@author: Chandra
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:51:29 2021

@author: Chandra
"""
#%% 1 Import Library
import pandas as pd # pemrosesan data csv
import string # Dipakai untuk manipulasi data array seperti menghapus tanda baca
import re # regular expression, manipulasi string
import nltk # Tokenisasi dan pencarian kata yang sering muncul
from nltk.tokenize import word_tokenize ## Dipakai untuk tokenisasi memisahkan kata dari kalimat

import matplotlib
#%% 2 Load Dataset
dataset = pd.read_csv("F:\Internship II\Data_Set\Sudah_di_label\datatest750.csv")
dataset = dataset[['Tweet','Nilai']]
#%%
dataseta = pd.read_csv("F:\Internship II\Data_Set\Sudah_di_label\datatest750.csv")
dataseta = dataseta[['Tweet','Nilai']]
#%% 3 Lowercase semua tweet

dataset["Tweet"] = dataset["Tweet"].str.lower()
#%% 4 Hapus angka pada tweet
dataset['Tweet'] = dataset['Tweet'].str.replace('\d+', '')

#%% hapus mention baru

dataset['Tweet'] = dataset['Tweet'].replace('@[A-Za-z0-9\S]+', '', regex=True)

#%% hapus hashtag

dataset['Tweet'] = dataset['Tweet'].replace('\#[\w\_]+', '', regex=True)
#%% 5 Menghapus tanda baca
dataset["Tweet"] = dataset['Tweet'].str.replace('[^\w\s]','')

#%% 6 Hapus whitespace

dataset['Tweet'].str.strip()
#%% 7 Hapus emoticon

dataset['Tweet'] = dataset['Tweet'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
#%% 8 Hapus Link
dataset['Tweet'] = dataset['Tweet'].str.replace('http\S+|www.\S+', '', case=False)
#%% 9 Hapus kata secara spesifik
dataset['Tweet'] = dataset['Tweet'].str.replace('dotcom', '')

#%% 10 hapus huruf berulang seperti hhhhh
min_threshold_rep = 3

dataset['Tweet']= dataset['Tweet'].str.replace(r'(\w)\1{%d,}'%(min_threshold_rep-1), r'\1')
#coba edit min_thereshold aga kata seperti tunggu tidak menjadi tungu

#%% 11 Translate kata bahasa inggris menjadi bahasa indonesia
from google_trans_new import google_translator  
translator = google_translator()  

    
dataset['Tweet'] = dataset['Tweet'].apply(translator.translate, lang_tgt='id')

#%% 12 Ubah kata gaul dan singkatan menjadi normal
kamus_slang = open('F:\Internship II\Wdictionaryupdatedv3.txt').read()
map_kamus_slang = {} #sebuah dictionary
list_kamus_slang = [] #sebuah list
for line in kamus_slang.split("\n"):
    if line != "":
        kamus = line.split("=")[0]
        kamus_luas = line.split("=")[1]
        list_kamus_slang.append(kamus)
        map_kamus_slang[kamus] = kamus_luas
list_kamus_slang = set(list_kamus_slang)

for word in list_kamus_slang:
    print(word)


def perubahan_kata_slang(tweet):
    new_text = []
    for w in tweet.split():
        if w.upper() in list_kamus_slang:
            new_text.append(map_kamus_slang[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

#coba = perubahan_kata_slang("gk ska vaksin")
dataset['Tweet'] = dataset['Tweet'].apply(perubahan_kata_slang)
#%% # 13 stopword adalah proses untuk menghapus kata yang tidak ada pengaruhnya seperti yg yang

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory # import stopword

factory = StopWordRemoverFactory() #import stopword
stopwords = factory.get_stop_words() #import stopword untuk mendapatkan list stopword bahasa indonesia
new_words=('kemdikbudri','hem','nih','yg','aja','fitrian','korantempo','tirtoid',
           'divisihumaspolri','polripresisi','poldasulsel','narasinewsroom','kompastv','thearieair',
           'detikcom','cnnindonesia','raiteguhparjaya','kompascom','yanadrn raditph','mah','pakahmadutomo',
           'riswokob','simpanthread','botkumpulan','motizenchannel','putrawadapi','hipohan','eemom','dhanangpuruhita',
           'cyjwj','indanasiaa','revanheryan','tyaszain','rodrichen','mrjn','raiteguhparjaya','korantempodigital',
           'korantempo','politikana','ismailfahmi','drpriono','penyejukhati','tuir','bhabinkamtibmas','btw') #menambahkan stopword baru
for i in new_words: #masukan stopword yang baru kedalam stopword bahasa indonesia
    stopwords.append(i)
print(stopwords)

stopword = factory.create_stop_word_remover() #gunakan function stopword remover untuk menghapus stopword dari dataframe

dataset['Tweet'] = dataset['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) 

#%% 14 Stemming = menyederhanakan kata seperti menghasilkan menjadi hasil
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # # import stemmer
factory = StemmerFactory() # import stemmer
stemmer = factory.create_stemmer() # # import stemmer

def stem_sentences(sentence):
    tokens = sentence.split() # variable tokens akan memisahkan setiap baris pada dataframe dataset
    stemmed_tokens = [stemmer.stem(token) for token in tokens] #variable stemmed_tokens berisi sebuah function yang
    #akan menyederhanakan setiap kata menjadi kata dasar misalkan semoga mejadi moga
    return ' '.join(stemmed_tokens) #jika setiap kata sudah selesai melalui proses stemming, return value

dataset['Tweet'] = dataset['Tweet'].apply(stem_sentences) #panggil function stem_sentences


#%% 15 Melihat kata yang sering ditemukan pada tweet
top_N=10
txt = dataset.Tweet.str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(txt)
word_dist = nltk.FreqDist(words)

stopwords = nltk.corpus.stopwords.words('indonesian')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 

print('Kata yang sering ditemukan:')
print('=' * 60)
rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N),
                    columns=['Kata', 'Frekuensi']).set_index('Kata')
print(rslt)
print('=' * 60)

matplotlib.style.use('ggplot')

rslt.plot.bar(width=0.2,rot=0)

#%%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = dataset['Tweet'].values 

wordcloud = WordCloud(width=680, height=480, max_words=50).generate(str(text))

plt.imshow(wordcloud)
plt.axis("off")
plt.show() 

#%%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Dropout

max_fatures = 100
#dibatasi 100 karena lebih baik menggunakan kata yang relevan atau sering muncul pada topik dibandingkan dengan 
#menggunakan seluruh kata yang hanya muncul sekalia atau dua kali
# =============================================================================
# #penjelasan tokenizer
# # num_words merupakan jumlah kata yang akan kita jadikan referensi, dalam hal ini kita akan menyimpan sebanyak
# # 100 kata unik untuk dijadikan referensi
# # Split merupakan pemisah, biasa diisi dengan -,= ataupun spasi
# =============================================================================
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(dataset['Tweet'].values)

#print("word_index : ",tokenizer.word_index) #melihat hasil dari fit_on_text

X = tokenizer.texts_to_sequences(dataset['Tweet'].values) #rubah tiap kata menjadi angka
X = pad_sequences(X) 
#print(X) array 2d

#X.ndim mengecek dimensi array
#pad_sequences digunakan untuk memastikan bahwa semua sequence di list mempunyai panjang yang sama
#dengan cara menambahkan 0 pada sequence yang kurang panjang misalkan
# pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]]) maka bagian yang diperbaiki oleh pad_sequences akan menjadi
# pad_sequences([[0, 1, 2, 3], [3, 4, 5, 6], [0, 0, 7, 8]])


#%% # membuat neural network LSTM
from keras.layers import Bidirectional
embed_dim = 128
lstm_out = 196


model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1])) #Merubah sequence menjadi vector
model.add(SpatialDropout1D(0.2)) #with this, model accuracy decerease
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(lstm_out,1)))
model.add(Bidirectional(LSTM(100)))
#model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#%%
from sklearn.model_selection import train_test_split

Y = pd.get_dummies(dataset['Nilai']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.0666, random_state = 50)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#%%
batch_size = 200
history = model.fit(X_train, Y_train, epochs =10, batch_size=batch_size, verbose =1)
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%

validation_size = 50

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

 #%%
print(len(X_test))
print(len(Y_test))
print(len(X_validate))  
print(len(Y_validate))

#%%

score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)
    
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

#%%
import numpy as np # linear algebra

twt = ['Warga ragu terhadap vaksin dr terawan yang tidak mengikuti standar WHO']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input

twt = pad_sequences(twt, maxlen=20, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")


#%%
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

#%%
import sklearn.metrics as metrics
y_pred = model.predict(X_train)

metrics.confusion_matrix(Y_train.argmax(axis=1), y_pred.argmax(axis=1))


#%%
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(metrics.confusion_matrix(Y_train.argmax(axis=1), y_pred.argmax(axis=1)), annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_head")
plt.ylabel("y_true")
plt.show()