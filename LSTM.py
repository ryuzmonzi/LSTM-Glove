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
import csv, re
def translator(user_string):
    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        fileName = "F:\Internship II\Wdictionaryupdatedv3 - copy.txt"
        # File Access mode [Read Mode]
        accessMode = "r"
        with open(fileName, accessMode) as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = row[1]
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    return ' '.join(user_string)
print('--------- Ganti kata singkatan -------------')
dataset['Tweet']=dataset['Tweet'].apply(lambda x: translator(x))
dataset["Tweet"] = dataset["Tweet"].str.lower()
print(dataset['Tweet'])
#%% # 13 stopword adalah proses untuk menghapus kata yang tidak ada pengaruhnya seperti yg yang

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory # import stopword

factory = StopWordRemoverFactory() #import stopword
stopwords = factory.get_stop_words() #import stopword untuk mendapatkan list stopword bahasa indonesia
new_words=('kemdikbudri','hem','nih','yg','aja','fitrian','korantempo','tirtoid',
           'divisihumaspolri','polripresisi','poldasulsel','narasinewsroom','kompastv','thearieair',
           'detikcom','cnnindonesia','raiteguhparjaya','kompascom','yanadrn raditph','mah','pakahmadutomo',
           'riswokob','simpanthread','botkumpulan','motizenchannel','putrawadapi','hipohan','eemom','dhanangpuruhita',
           'cyjwj','indanasiaa','revanheryan','tyaszain','rodrichen','mrjn','raiteguhparjaya','korantempodigital',
           'korantempo','politikana','ismailfahmi','drpriono','penyejukhati','tuir','bhabinkamtibmas','btw','nya',
           'aaron','fpan','lang','la','jk','bal','as','m','ikawati','zullies','ap','ancoo','huduk','kucur','unk',
           'dirumbling','wargar','nihdan','alesha','papa','zik','agam','simbolon','effendi','abjrizal','imanuel',
           'o','arifin','indknesia','samael','twk','johnny','bahuri','yadia','marta','firli','amapantatara','tgh',
           'wgh','jamaluddin','yekar','khairy','lp','yaolo','y','kagan','apemanya','lisa','tum','nt','lisamartna',
           'nk','hazer','kamaruddin','my','palsulama','dgr','ayu','bgm','lae','v','cs','putihmungkin','halalampmematikanresiko',
           'amanamptak','vcinasinopharmsinovaccansinomasih','u','gtlt','w','rampd','bajer','uwuy','jajajah','asama','pny','poldasu',
           'sljj','ubn','bur','rowdome','bla','trikus','smh','z','al','cak','ali','q','ndoro') #menambahkan stopword baru
for i in new_words: #masukan stopword yang baru kedalam stopword bahasa indonesia
    stopwords.append(i)
#print(stopwords)

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

print(words[5231])

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
from glove import Corpus, Glove
from nltk.corpus import gutenberg
from multiprocessing import Pool
from scipy import spatial
import pandas as pd
import nltk

tweetText = dataset['Tweet']
tweetText = tweetText.apply(word_tokenize)
tweetText.head()

corpus = Corpus()
corpus.fit(tweetText, window=2)   # window parameter denotes the distance of context
glove = Glove(no_components=50, learning_rate=0.01)
 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model') 




with open("results_glove.txt", "w") as f:
    for word in glove.dictionary:
        f.write(word)
        f.write(" ")
        for i in range(0, 5):
            f.write(str(glove.word_vectors[glove.dictionary[word]][i]))
            f.write(" ")
        f.write("\n")

glove.most_similar('aneh')
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
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

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
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))
#print("word_index : ",tokenizer.word_index) #melihat hasil dari fit_on_text

X = tokenizer.texts_to_sequences(dataset['Tweet'].values) #rubah tiap kata menjadi angka
X = pad_sequences(X) 

#print(X) array 2d

#X.ndim mengecek dimensi array
#pad_sequences digunakan untuk memastikan bahwa semua sequence di list mempunyai panjang yang sama
#dengan cara menambahkan 0 pada sequence yang kurang panjang misalkan
# pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]]) maka bagian yang diperbaiki oleh pad_sequences akan menjadi
# pad_sequences([[0, 1, 2, 3], [3, 4, 5, 6], [0, 0, 7, 8]])


#%%
from sklearn.model_selection import train_test_split

Y = pd.get_dummies(dataset['Nilai']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.0666, random_state = 50)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
print("test set size " + str(len(X_test)))

#%% Atur EMbedding
import numpy as np # linear algebra
EMBEDDING_DIM = 5      # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "F:\Internship II\\results_glove"+str(EMBEDDING_DIM)+".txt"


embeddings_index = {}
f = open(GLOVE_DIR)
print('Loading GloVe from:', GLOVE_DIR,'...', end='')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print("Done.\n Proceeding with Embedding Matrix...", end="")

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(" Completed!")
#%% # membuat neural network LSTM
num_words = min(max_fatures, len(word_index)) + 1
from keras.initializers import Constant
embedding_dim = 100
sequence_length = X.shape[1]
lstm_out = 196

model = Sequential()
model.add(Embedding(643,
                    20,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=sequence_length,
                    trainable=True))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)) #0.2 pada dropout artinya 20% dari layer akan di drop 
# dan lstm_out mendefinisikan ukuran dari output yang akan dikeluarkan oleh LSTM
model.add(Dense(2,activation='softmax')) 
#ada 3 jenis, softmax, sigmoid, dan relu. softmax dipakai karena memberikan akurasi lebih tinggi dibandingkan dengan yang lain
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


#%%
batch_size = 200
history = model.fit(X_train, Y_train, epochs =40, batch_size=batch_size, verbose = 1)
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


from sklearn.metrics import accuracy_score
score_lstm= accuracy_score(X_test,Y_test)

from sklearn.metrics import classification_report
print(classification_report(X_validate,Y_validate))
#%%


twt = ['warga ragu untuk ikut vaksin']
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