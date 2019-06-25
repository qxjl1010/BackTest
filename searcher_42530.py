import math
import re
import datetime
import socketserver
import threading
import json

from gensim import corpora, models, similarities
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import pandas as pd
import mysql.connector

from timeit import default_timer as timer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop = set(stopwords.words('english'))

headlines_list = []
raw_headlines_list = []
time_list = []
ID_list = []
keep_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']

# My SQL Credentials--------------------------------------------------------
mysql_user = 'root'
mysql_password = 'Bitcoin2018!'
mysql_database = 'cryptote_db2'
mysql_host = '35.203.23.69' #Prod

time_format = '%Y-%m-%d %H:%M:%S'

onemonth_headline_list = []
onemonth_time_list = []
onemonth_ID_list =[]
created_at_list = []

cnx = mysql.connector.connect(user = mysql_user, password = mysql_password, host = mysql_host, database = mysql_database )
crsr = cnx.cursor(buffered=True)

# LAST_UPDATE = timer()

rst_dict = {}

def get_initial_list():
    global cnx
    global crsr
    # Now = datetime.datetime.now()
    # new: read from DB
    # get sixhours
    # onemonth_before = Now + datetime.timedelta(days=-30)
    # str_onemonth_before = onemonth_before.strftime(time_format)



    print("start building list")
    print("fetching from DB...")

    headlines_fromDB = []

    if cnx.is_connected():
        pass
    else:
        cnx = mysql.connector.connect(user = mysql_user, password = mysql_password, host = mysql_host, database = mysql_database )
    crsr = cnx.cursor(buffered=True)
        
    command = "select id,text,created_at from headline_archives where author != 'Crypto Terminal AI' and topic != 'hashtag' and two_days != 0"

    crsr.execute(command)


    rows = crsr.fetchall()

    headlines_fromDB += rows

        

    crsr.close()

    print("list length:")
    print(len(headlines_fromDB))

    print("fetch successful!")
    print("building list...")
    crsr.close()
    cnx.close()





    for headline in headlines_fromDB:
        clean_headline = ""
        word_list = headline[1].split()
        for word in word_list:
            if word[0] == '@' or word[0] == '#' or word[0] == '&':
                continue
            c_word = ''
            for w in word:
                if w in keep_list:
                    c_word += w
            word = c_word.replace(',', '')
            word = word.replace('?', '')
            word = word.replace('.', '')
            word = word.replace("'", '')
            word = word.replace("\"", '')
            word = word.replace('-', '')
            if word != '':
                clean_word = lemmatizer.lemmatize(word)
                # clean_word = stemmer.stem(clean_word)
                clean_word = word.lower()
                if clean_word in stop:
                    continue
            else:
                continue                
            clean_headline = clean_headline + clean_word + " "
        clean_headline_list = clean_headline.split()
        if len(clean_headline_list) <= 3:
            continue
        created_at = headline[2]
        t = created_at.strftime(time_format)
        created_at_list.append(t)
        onemonth_headline_list.append(headline[1])
        onemonth_time_list.append(t)
        onemonth_ID_list.append(headline[0])



    for line in onemonth_headline_list:
        clean_headline = ""
        word_list = line.split()
        for word in word_list:
            if word[0] == '@' or word[0] == '#' or word[0] == '&':
                continue
            c_word = ''
            for w in word:
                if w in keep_list:
                    c_word += w
            word = c_word.replace(',', '')
            word = word.replace('?', '')
            word = word.replace('.', '')
            word = word.replace("'", '')
            word = word.replace("\"", '')
            word = word.replace('-', '')
            if word != '':
                clean_word = lemmatizer.lemmatize(word)
                # clean_word = stemmer.stem(clean_word)
                clean_word = word.lower()
                if clean_word in stop:
                    continue
            else:
                continue                
            clean_headline = clean_headline + clean_word + " "
        headlines_list.append(clean_headline[:-1])
        raw_headlines_list.append(line)
    print("headlines list building done!")

    for line in onemonth_time_list:
        time_list.append(line)
    print("time list building done!")

    for line in onemonth_ID_list:
        ID_list.append(line)
    print("ID list building done!")


# bm25
f = []
tf = {}
idf = {} 
k1 = 1.5
b = 0.75
def inition(docs):
    D = len(docs)
    avgdl = sum([len(doc)+ 0.0 for doc in docs]) / D
    for doc in docs:
        tmp = {}
        for word in doc:
            tmp[word] = tmp.get(word, 0) + 1 
        f.append(tmp)
        for k in tmp.keys():
            tf[k] = tf.get(k, 0) + 1
    for k, v in tf.items():
        idf[k] = math.log(D - v + 0.5) - math.log(v + 0.5)
    return D, avgdl

def sim(doc, index):
    score = 0.0
    for word in doc:
        if word not in f[index]:
            continue
        d = len(headlines_list[index])
        score += (idf[word] * f[index][word] * (k1 + 1) / (f[index][word] + k1 * (1 - b + b * d / avgdl)))
    return score

def simall(doc):
    scores = []
    i = 0
    for index in range(D):
        score = sim(doc, index)
        tmp_tuple = (i, score)
        scores.append(tmp_tuple)
        i += 1
    scores = sorted(scores, key=lambda x:x[1], reverse=True)
    return scores

def get_sentences(doc):
    line_break = re.compile('[\r\n]')
    delimiter = re.compile('[，。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
    return sentences
get_initial_list()
raw_words_dict = [word.split() for word in raw_headlines_list]
words_dict = [word.split() for word in headlines_list]
D,avgdl = inition(words_dict)





# tf-idf


# get_initial_list()
# raw_words_dict = [word.split() for word in raw_headlines_list]
# words_dict = [word.split() for word in headlines_list]
print("building dictionary...")
dictionary = corpora.Dictionary(words_dict)
dictionary.token2id
corpus = [dictionary.doc2bow(doc) for doc in words_dict]
tfidf = models.TfidfModel(corpus)
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))

print("building done!")
print("please input headline:")






class ThreadedTCPRequestHandler( socketserver.BaseRequestHandler ) :
    def handle( self ) :
        cnx = mysql.connector.connect(user = mysql_user, password = mysql_password, host = mysql_host, database = mysql_database )
        crsr = cnx.cursor(buffered=True)
        # LAST_UPDATE = timer()
        # while 1==1:


        if True:
            try :
                send_dict = {}
                self.data = self.request.recv( 10240 ).strip()
                # print ("--> %s wrote:\n%s" % ( self.client_address[ 0 ], str( self.data ) ))
                
                # update constantly, block for now
                '''
                NEW_UPDATE = timer()
                print("NEW is:")
                print(NEW_UPDATE)
                print("LAST is:")
                print(LAST_UPDATE)
                if NEW_UPDATE - LAST_UPDATE >= 604800:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("Waiting for Update")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    get_initial_list()
                    print("Update complete")
                    LAST_UPDATE = timer()   
                '''
                

                test_headline = str(self.data)

                if test_headline == "":
                    self.request.send( bytes( "[]", encoding = "utf8") )
                    return

                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("START TIME:")
                print(datetime.datetime.now())
                clean_headline = ""
                for word in test_headline.split():
                    if word[0] == '@' or word[0] == '#' or word[0] == '&':
                        continue
                    c_word = ''
                    for w in word:
                        if w in keep_list:
                            c_word += w
                    word = c_word.replace(',', '')
                    word = word.replace('?', '')
                    word = word.replace('.', '')
                    word = word.replace("'", '')
                    word = word.replace("\"", '')
                    word = word.replace('-', '')
                    if word != '':
                        clean_word = lemmatizer.lemmatize(word)
                        # clean_word = stemmer.stem(clean_word)
                        clean_word = word.lower()
                        if clean_word in stop:
                            continue        
                        clean_headline = clean_headline + clean_word + " "
                headline_list = clean_headline.split()
                
                test_vec = dictionary.doc2bow(headline_list)

                tf_idf_sim = index[tfidf[test_vec]]

                tf_idf_rst = sorted(enumerate(tf_idf_sim), key = lambda item: -item[1])
                
                print("===============================================================================================================")
                print("AFTER CLEAN:")
                print(datetime.datetime.now())
                print("===============================================================================================================")

                # bm25_rst = simall(headline_list)

                # print("TF_IDF result:")
                # print("===============================================================================================================")
                '''
                column = ["headline","data","ID"]
                row = ["rank1","rank2","rank3","rank4","rank5"]
                list2csv = []
                '''
                last_headline = ""
                i = 0
                rank = 1
                tf_idf_list_sortByscore = []

                enhanced_list = []
                # print("length tf_idf_rst is:")
                # print(len(tf_idf_rst))
                while True:
                    if tf_idf_rst[i][1] < 0.1:
                        break
                    clean_headline = ""
                    multiple = 1
                    for word in raw_words_dict[tf_idf_rst[i][0]]:
                        if word[0] == '@' or word[0] == '#' or word[0] == '&':
                            continue
                        c_word = ''
                        for w in word:
                            if w in keep_list:
                                c_word += w
                        word = c_word.replace(',', '')
                        word = word.replace('?', '')
                        word = word.replace('.', '')
                        word = word.replace("'", '')
                        word = word.replace("\"", '')
                        word = word.replace('-', '')
                        if word != '':
                            clean_word = lemmatizer.lemmatize(word)
                            # clean_word = stemmer.stem(clean_word)
                            clean_word = word.lower()
                            if clean_word in stop:
                                continue        
                            clean_headline = clean_headline + clean_word + " "
                    top_headline_list = clean_headline.split()
                    # check first 3 words
                    if headline_list[0] in top_headline_list[0:3]:
                        multiple += 1
                    if headline_list[1] in top_headline_list[0:3]:
                        multiple += 1
                    if headline_list[2] in top_headline_list[0:3]:
                        multiple += 1
                    weight = tf_idf_rst[i][1] * multiple
                    tmp_list = [tf_idf_rst[i][0], weight]
                    enhanced_list.append(tmp_list)
                    i += 1
                sorted_enhanced_list = sorted(enhanced_list, key = lambda item: -item[1])                    
                # print("length of sorted_enhanced_list is:")
                # print(len(sorted_enhanced_list))
                
                # print("###############################################################################################################")
                # print("top score:")
                # print("###############################################################################################################")
                print("===============================================================================================================")
                print("AFTER GET:")
                print(datetime.datetime.now())
                print("===============================================================================================================")
                i = 0
                while True:
                    i += 1
                    if rank > 5:
                        break        
                    headline = ""
                    for j in range(len(raw_words_dict[sorted_enhanced_list[i-1][0]])):
                        if raw_words_dict[sorted_enhanced_list[i-1][0]][j][0] in ['@','#','&']:
                            continue
                        clean_word = raw_words_dict[sorted_enhanced_list[i-1][0]][j].replace("\"","")
                        headline += (clean_word + " ")
                    if last_headline == headline:
                        continue
                    print("---------------------------------------------------------------------------------------------------------------")
                    print("BEFORE GET 5::")
                    print(datetime.datetime.now())
                    print("---------------------------------------------------------------------------------------------------------------")
                    # get 5 mins price change
                    request_command = "select five_mins from headline_archives where id = " + str(ID_list[sorted_enhanced_list[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    five_mins_price_change = price_change_list[0][0]
                    print("---------------------------------------------------------------------------------------------------------------")
                    print("BEFORE GET 30::")
                    print(datetime.datetime.now())
                    print("---------------------------------------------------------------------------------------------------------------")
                    # get 30 mins price change
                    request_command = "select thirty_mins from headline_archives where id = " + str(ID_list[sorted_enhanced_list[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    thirty_mins_price_change = price_change_list[0][0]
                    print("---------------------------------------------------------------------------------------------------------------")
                    print("BEFORE GET 60::")
                    print(datetime.datetime.now())
                    print("---------------------------------------------------------------------------------------------------------------")
                    # get 60 mins price change
                    request_command = "select sixty_mins from headline_archives where id = " + str(ID_list[sorted_enhanced_list[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    sixty_mins_price_change = price_change_list[0][0]
                    print("---------------------------------------------------------------------------------------------------------------")
                    print("BEFORE GET 2 days::")
                    print(datetime.datetime.now())
                    print("---------------------------------------------------------------------------------------------------------------")
                    # get 2 days price change
                    request_command = "select two_days from headline_archives where id = " + str(ID_list[sorted_enhanced_list[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    two_days_price_change = price_change_list[0][0]            
                    print("---------------------------------------------------------------------------------------------------------------")
                    print("AFTER GET 2 days::")
                    print(datetime.datetime.now())
                    print("---------------------------------------------------------------------------------------------------------------")


                    headline_dict = {}
                    last_headline = headline


                    headline_dict['rank'] = rank
                    rank += 1       
                    headline_dict['content'] = headline[:-1]
                    
                    headline_dict['prices'] = {'5mins':five_mins_price_change, '30mins':thirty_mins_price_change, '60mins':sixty_mins_price_change, '1day':two_days_price_change}
                    
                    headline_dict['score'] = str(sorted_enhanced_list[i-1][1])

                    
                    headline_dict['time'] = time_list[sorted_enhanced_list[i-1][0]]
                    headline_dict['ID'] = ID_list[sorted_enhanced_list[i-1][0]]
                    # print("======================================")
                    
                    # list_tmp = [headline[:-1],time_list[sorted_enhanced_list[i-1][0]], ID_list[sorted_enhanced_list[i-1][0]]]
                    # list2csv.append(list_tmp)
                    tf_idf_list_sortByscore.append(headline_dict)

                send_dict["tf_idf_sortByScore"] = tf_idf_list_sortByscore
                

                '''
                print("###############################################################################################################")
                print("top price change:")
                print("###############################################################################################################")
                last_headline = ""
                i = 0
                all_price_change_list = []
                while True:
                    i += 1
                    if tf_idf_rst[i-1][1] < 0.25:
                        break
                    headline = ""
                    for j in range(len(raw_words_dict[tf_idf_rst[i-1][0]])):
                        headline += (raw_words_dict[tf_idf_rst[i-1][0]][j] + " ")
                    if last_headline == headline:
                        continue
                    
                    # get 5 mins price change
                    request_command = "select five_mins from headline_archives where id = " + str(ID_list[tf_idf_rst[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    five_mins_price_change = price_change_list[0][0]

                    # get 30 mins price change
                    request_command = "select thirty_mins from headline_archives where id = " + str(ID_list[tf_idf_rst[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    thirty_mins_price_change = price_change_list[0][0]

                    # get 60 mins price change
                    request_command = "select sixty_mins from headline_archives where id = " + str(ID_list[tf_idf_rst[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    sixty_mins_price_change = price_change_list[0][0]

                    last_headline = headline
                    tmp_list = [headline[:-1], five_mins_price_change, thirty_mins_price_change, sixty_mins_price_change, tf_idf_rst[i-1][1],time_list[tf_idf_rst[i-1][0]],ID_list[tf_idf_rst[i-1][0]],created_at_list[tf_idf_rst[i-1][0]]]
                    all_price_change_list.append(tmp_list)
                
                total_nums = len(all_price_change_list)

                tf_idf_rst_sortPrice = sorted(all_price_change_list, key = lambda item:abs(item[3]), reverse=True)
                # i = 0
                rank = 1
                tf_idf_list_sortByprice = []
                print("list length:")
                print(total_nums)
                for i in range(total_nums):
                    if i >= 5:
                        break
                    headline_dict = {}
                    print("======================================")
                    headline_dict["rank"] = rank
                    print("rank"+str(rank)+":")
                    rank += 1          
                    print(tf_idf_rst_sortPrice[i][0])
                    headline_dict["content"] = tf_idf_rst_sortPrice[i][0]
                    
                    # print("price change:"+str(tf_idf_rst_sortPrice[i][1]))
                    headline_dict['prices'] = {'5mins':tf_idf_rst_sortPrice[i][1], '30mins':tf_idf_rst_sortPrice[i][2], '60mins':tf_idf_rst_sortPrice[i][3], '1day':'0'}
                    print("price change:")
                    print(headline_dict['prices'])
                    print("score:"+str(tf_idf_rst_sortPrice[i][4]))
                    headline_dict['score'] = str(tf_idf_rst_sortPrice[i][4])

                    print("time:"+tf_idf_rst_sortPrice[i][5])
                    headline_dict['time'] = tf_idf_rst_sortPrice[i][5]

                    print("ID:"+str(tf_idf_rst_sortPrice[i][6]))
                    headline_dict['ID'] = tf_idf_rst_sortPrice[i][6]
                    print("======================================")   
                    i += 1 
                    tf_idf_list_sortByprice.append(headline_dict)
                send_dict["tf_idf_sortByPrice"] = tf_idf_list_sortByprice


                
                tf_idf_rst_sortTime = sorted(all_price_change_list, key = lambda item:(item[5]))
                print("###############################################################################################################")
                print("time(old to new):")
                print("###############################################################################################################")    
                i = 0
                rank = 1
                tf_idf_list_sortBytime_OldtoNew = []
                while True:
                    if i >= 5:
                        break
                    headline_dict = {}
                    print("======================================")
                    headline_dict["rank"] = rank
                    print("rank"+str(rank)+":")  
                    rank += 1        
                    print(tf_idf_rst_sortTime[i][0])
                    headline_dict["content"] = tf_idf_rst_sortTime[i][0]
                    # print("price change:"+str(tf_idf_rst_sortTime[i][1]))
                    headline_dict['prices'] = {'5mins':tf_idf_rst_sortTime[i][1], '30mins':tf_idf_rst_sortTime[i][2], '60mins':tf_idf_rst_sortTime[i][3]}
                    print("price change:")
                    print(headline_dict['prices'])
                    print("score:"+str(tf_idf_rst_sortTime[i][4]))
                    headline_dict['score'] = str(tf_idf_rst_sortTime[i][4])
                    print("time:"+tf_idf_rst_sortTime[i][5])
                    headline_dict['time'] = tf_idf_rst_sortTime[i][5]
                    print("ID:"+str(tf_idf_rst_sortTime[i][6]))
                    headline_dict['ID'] = tf_idf_rst_sortTime[i][6]
                    print("======================================")   
                    i += 1 
                    tf_idf_list_sortBytime_OldtoNew.append(headline_dict)
                send_dict["tf_idf_sortByTime_oldToNew"] = tf_idf_list_sortBytime_OldtoNew


                tf_idf_rst_sortTimeReverse = sorted(all_price_change_list, key = lambda item:(item[5]), reverse=True)
                print("###############################################################################################################")
                print("time(new to old):")
                print("###############################################################################################################")    
                i = 0
                rank = 1
                tf_idf_list_sortBytime_NewtoOld = []
                while True:
                    if i >= 5:
                        break
                    headline_dict = {}    
                    print("======================================")
                    print("rank"+str(rank)+":")
                    headline_dict["rank"] = rank
                    rank += 1          
                    print(tf_idf_rst_sortTimeReverse[i][0])
                    headline_dict["content"] = tf_idf_rst_sortTimeReverse[i][0]
                    # print("price change:"+str(tf_idf_rst_sortTimeReverse[i][1]))
                    headline_dict['prices'] = {'5mins':tf_idf_rst_sortTimeReverse[i][1], '30mins':tf_idf_rst_sortTimeReverse[i][2], '60mins':tf_idf_rst_sortTimeReverse[i][3]}
                    print("price change:")
                    print(headline_dict['prices'])
                    print("score:"+str(tf_idf_rst_sortTimeReverse[i][4]))
                    headline_dict['score'] = str(tf_idf_rst_sortTimeReverse[i][4])
                    print("time:"+tf_idf_rst_sortTimeReverse[i][5])
                    headline_dict['time'] = tf_idf_rst_sortTimeReverse[i][5]
                    print("ID:"+str(tf_idf_rst_sortTimeReverse[i][6]))
                    headline_dict['ID'] = tf_idf_rst_sortTimeReverse[i][6]
                    print("======================================")   
                    i += 1    
                    tf_idf_list_sortBytime_NewtoOld.append(headline_dict)
                send_dict["tf_idf_sortByTime_newToOld"] = tf_idf_list_sortBytime_NewtoOld     


                print("===============================================================================================================\n\n")
                print("===============================================================================================================")

                print("BM25 result:")
                print("===============================================================================================================")
                list2csv = []
                last_headline = ""
                i = 0
                rank = 1
                BM25_list_sortByscore = []

                print("###############################################################################################################")
                print("top score:")
                print("###############################################################################################################")
                while True:    
                    i += 1
                    if rank > 5:
                        break    
                    headline = ""
                    for j in range(len(raw_words_dict[bm25_rst[i][0]])):
                        headline += (raw_words_dict[bm25_rst[i][0]][j] + " ")
                    if last_headline == headline:
                        continue

                    # get 5 mins price change
                    request_command = "select five_mins from headline_archives where id = " + str(ID_list[tf_idf_rst[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    five_mins_price_change = price_change_list[0][0]

                    # get 30 mins price change
                    request_command = "select thirty_mins from headline_archives where id = " + str(ID_list[tf_idf_rst[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    thirty_mins_price_change = price_change_list[0][0]

                    # get 60 mins price change
                    request_command = "select sixty_mins from headline_archives where id = " + str(ID_list[tf_idf_rst[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    sixty_mins_price_change = price_change_list[0][0]

                    headline_dict = {}
                    last_headline = headline
                    print("======================================")
                    print("rank"+str(rank)+":")
                    headline_dict["rank"] = rank
                    rank += 1
                    print(headline[:-1])
                    headline_dict["content"] = headline[:-1]
                    # print("price change:"+str(price_change))
                    headline_dict['prices'] = {'5mins':five_mins_price_change, '30mins':thirty_mins_price_change, '60mins':sixty_mins_price_change}
                    print("price change:")
                    print(headline_dict['prices'])
                    print("score:"+str(bm25_rst[i][1]))
                    headline_dict['score'] = str(bm25_rst[i-1][1])
                    print("time:"+time_list[bm25_rst[i-1][0]])
                    headline_dict['time'] = time_list[bm25_rst[i-1][0]]
                    print("ID:"+str(ID_list[bm25_rst[i-1][0]]))
                    headline_dict['ID'] = ID_list[bm25_rst[i-1][0]]
                    print("======================================")        
                    # list_tmp = [headline[:-1],time_list[bm25_rst[i-1][0]], ID_list[bm25_rst[i-1][0]]]
                    # list2csv.append(list_tmp)
                    BM25_list_sortByscore.append(headline_dict)
                send_dict["BM25_sortByScore"] = BM25_list_sortByscore


                print("###############################################################################################################")
                print("top price change:")
                print("###############################################################################################################")
                last_headline = ""
                i = 0
                all_price_change_list = []
                while True:
                    i += 1
                    if bm25_rst[i-1][1] < 4:
                        break
                    headline = ""
                    for j in range(len(raw_words_dict[bm25_rst[i-1][0]])):
                        headline += (raw_words_dict[bm25_rst[i-1][0]][j] + " ")
                    if last_headline == headline:
                        continue

                    # get 5 mins price change    
                    request_command = "select five_mins from headline_archives where id = " + str(ID_list[bm25_rst[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    five_mins_price_change = price_change_list[0][0]

                    # get 30 mins price change
                    request_command = "select thirty_mins from headline_archives where id = " + str(ID_list[bm25_rst[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    thirty_mins_price_change = price_change_list[0][0]

                    # get 60 mins price change
                    request_command = "select sixty_mins from headline_archives where id = " + str(ID_list[bm25_rst[i-1][0]])
                    crsr.execute(request_command)
                    price_change_list = crsr.fetchall()
                    sixty_mins_price_change = price_change_list[0][0]


                    last_headline = headline
                    tmp_list = [headline[:-1], five_mins_price_change, thirty_mins_price_change, sixty_mins_price_change, bm25_rst[i-1][1],time_list[bm25_rst[i-1][0]],ID_list[bm25_rst[i-1][0]],created_at_list[bm25_rst[i-1][0]]]
                    all_price_change_list.append(tmp_list)
                
                bm25_rst_sortPrice = sorted(all_price_change_list, key = lambda item:abs(item[1]), reverse=True)
                i = 0
                rank = 1
                BM25_list_sortByPrice = []
                while True:
                    headline_dict = {}  
                    if i >= 5:
                        break
                    print("======================================")
                    print("rank"+str(rank)+":")
                    headline_dict["rank"] = rank
                    rank += 1          
                    print(bm25_rst_sortPrice[i][0])
                    headline_dict["content"] = bm25_rst_sortPrice[i][0]
                    # print("price change:"+str(bm25_rst_sortPrice[i][1]))
                    headline_dict['prices'] = {'5mins':bm25_rst_sortPrice[i][1], '30mins':bm25_rst_sortPrice[i][2], '60mins':bm25_rst_sortPrice[i][3]}
                    print("price change:")
                    print(headline_dict['prices'])
                    print("score:"+str(bm25_rst_sortPrice[i][4]))
                    headline_dict['score'] = str(bm25_rst_sortPrice[i][4])
                    print("time:"+bm25_rst_sortPrice[i][5])
                    headline_dict['time'] = bm25_rst_sortPrice[i][5]
                    print("ID:"+str(bm25_rst_sortPrice[i][6]))
                    headline_dict['ID'] = bm25_rst_sortPrice[i][6]
                    print("======================================")   
                    i += 1 
                    BM25_list_sortByPrice.append(headline_dict)
                send_dict["BM25_sortByPrice"] = BM25_list_sortByPrice

                bm25_rst_sortTime = sorted(all_price_change_list, key = lambda item:(item[5]))
                print("###############################################################################################################")
                print("time(old to new):")
                print("###############################################################################################################")    
                i = 0
                rank = 1
                BM25_list_sortByTime_oldToNew = []
                while True:
                    if i >= 5:
                        break
                    print("======================================")
                    print("rank"+str(rank)+":")         
                    headline_dict["rank"] = rank
                    rank += 1 
                    print(bm25_rst_sortTime[i][0])
                    headline_dict["content"] = bm25_rst_sortTime[i][0]
                    #print("price change:"+str(bm25_rst_sortTime[i][1]))
                    headline_dict['prices'] = {'5mins':bm25_rst_sortTime[i][1], '30mins':bm25_rst_sortTime[i][2], '60mins':bm25_rst_sortTime[i][3]}
                    print("price change:")
                    print(headline_dict['prices'])
                    print("score:"+str(bm25_rst_sortTime[i][4]))
                    headline_dict['score'] = str(bm25_rst_sortTime[i][4])
                    print("time:"+bm25_rst_sortTime[i][5])
                    headline_dict['time'] = bm25_rst_sortTime[i][5]
                    print("ID:"+str(bm25_rst_sortTime[i][6]))
                    headline_dict['ID'] = bm25_rst_sortTime[i][6]
                    print("======================================")   
                    i += 1 
                    BM25_list_sortByTime_oldToNew.append(headline_dict)
                send_dict["BM25_sortByTime_oldToNew"] = BM25_list_sortByTime_oldToNew


                bm25_rst_sortTimeReverse = sorted(all_price_change_list, key = lambda item:(item[5]), reverse=True)
                print("###############################################################################################################")
                print("time(new to old):")
                print("###############################################################################################################")    
                i = 0
                rank = 1
                BM25_list_sortByTime_newToOld = []
                while True:
                    if i >= 5:
                        break
                    print("======================================")
                    print("rank"+str(rank)+":")       
                    headline_dict["rank"] = rank
                    rank += 1    
                    print(bm25_rst_sortTimeReverse[i][0])
                    headline_dict["content"] = bm25_rst_sortTimeReverse[i][0]
                    # print("price change:"+str(bm25_rst_sortTimeReverse[i][1]))
                    headline_dict['prices'] = {'5mins':bm25_rst_sortTimeReverse[i][1], '30mins':bm25_rst_sortTimeReverse[i][2], '60mins':bm25_rst_sortTimeReverse[i][3]}
                    print("price change:")
                    print(headline_dict['prices'])
                    print("score:"+str(bm25_rst_sortTimeReverse[i][4]))
                    headline_dict['score'] = str(bm25_rst_sortTimeReverse[i][4])
                    print("time:"+bm25_rst_sortTimeReverse[i][5])
                    headline_dict['time'] = bm25_rst_sortTimeReverse[i][5]
                    print("ID:"+str(bm25_rst_sortTimeReverse[i][6]))
                    headline_dict['ID'] = bm25_rst_sortTime[i][6]
                    print("======================================")   
                    i += 1 
                    BM25_list_sortByTime_newToOld.append(headline_dict)
                send_dict["BM25_sortByTime_newToOld"] = BM25_list_sortByTime_newToOld
                '''
                
                print("===============================================================================================================")
                print("END TIME:")
                print(datetime.datetime.now())
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


                # send the final result
                self.request.send( bytes( "%s\n" % str( json.dumps( tf_idf_list_sortByscore )), encoding = "utf8") )
                
            except Exception as inst:
                with open("error.log", 'a') as outfile:
                    print(type(inst))    # the exception instance
                    print(inst.args)     # arguments stored in .args
                    print(inst)  
                    outfile.write(type(inst))
                    outfile.write(inst.args)
                    outfile.write(inst)
                outfile.close()
            self.request.close()



class ThreadedTCPServer( socketserver.ThreadingMixIn, socketserver.TCPServer ) :
    pass

if __name__ == "__main__" :
    try:
        HOST = ""
        PORT = 42530
        server = ThreadedTCPServer( ( HOST, PORT ), ThreadedTCPRequestHandler )
        server.serve_forever()
        server_thread = threading.Thread( target=server.serve_forever )
        server_thread.setDaemon( True )
        server_thread.start()
        while True :
            pass
    except Exception as inst:
        with open("error.log", 'a') as outfile:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)  
            outfile.write(type(inst))
            outfile.write(inst.args)
            outfile.write(inst)
        outfile.close()

