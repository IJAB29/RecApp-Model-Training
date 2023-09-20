import pandas as pd
import numpy as np
import string
from functools import reduce
from nltk import word_tokenize
from nltk import PorterStemmer
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from gensim import corpora, models

# import nltk
# nltk.download("wordnet")
# nltk.download('omw-1.4')
# nltk.download("punkt")
# nltk.download("stopwords")

stop_words = stopwords.words("english")
custom = ["also", "would", "will"]
stop_words.extend(custom)
p_stemmer = PorterStemmer()
wnl = WordNetLemmatizer()


# def cleanText(text):
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     text = text.lower()
#
#     return text


def lemmaBalls(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def getTokens(text, lem=True):
    text = str(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    print("Remove punct:", text)
    text = text.lower()
    print("Lower case:", text)
    tokens = word_tokenize(text)
    print("Tokenize:", tokens)
    # tokens = [t for t in tokens if t not in stop_words and len(t) > 3]
    # tokens = [lemmaBalls(t) for t in tokens if t not in stop_words and len(t) > 3]
    if lem:
        # tokens = [p_stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 3]
        remove_stopw = demoStopwordsRemove(tokens)
        print("NO SW:", remove_stopw)
        filter_byl = demoFilter(remove_stopw)
        print("FILTER:", filter_byl)
        stemmed = demoStem(filter_byl)
        print("STEMMED:", stemmed)
    else:
        tokens = [t for t in tokens if t not in stop_words and len(t) > 3]

    return tokens

def demoFilter(tokens):
    return [t for t in tokens if len(t) > 3]

def demoStopwordsRemove(tokens):
    return [t for t in tokens if t not in stop_words]

def demoStem(tokens):
    return [p_stemmer.stem(t) for t in tokens]


def prepTerms(doc_label, doc_texts):
    terms = defaultdict(list)

    for n in range(0, len(doc_texts)):
        label = doc_label[n]
        text = doc_texts[n]
        # text = cleanText(text)
        tokens = getTokens(text)
        terms[label].extend(tokens)

    return terms


def getFD(terms):
    """dictionary containing a list of tuples (term, frequency)
    {Capstone: [(system, 42), ()], Thesis: [(), ()]}"""
    distributions = defaultdict(list)

    for label, tokens in terms.items():
        fd = FreqDist(tokens)
        distributions[label].extend(fd.most_common())

    return distributions


def getFrequencyDF(frequency_distribution):
    """separate term and value from distributions"""
    capstone_dict = {item[0]: item[1] for item in frequency_distribution["Capstone"]}
    thesis_dict = {item[0]: item[1] for item in frequency_distribution["Thesis"]}
    feasibility_dict = {item[0]: item[1] for item in frequency_distribution["Feasibility Study"]}
    case_dict = {item[0]: item[1] for item in frequency_distribution["Case Study"]}

    """unify the terms from each class"""
    unique_words = reduce(np.union1d, (np.array(list(capstone_dict.keys())), np.array(list(thesis_dict.keys())),
                                       np.array(list(feasibility_dict.keys())), np.array(list(case_dict.keys()))))

    """initialize shape of the arrays"""
    capstone_freq = np.zeros(len(unique_words))
    thesis_freq = np.zeros(len(unique_words))
    case_freq = np.zeros(len(unique_words))
    feasibility_freq = np.zeros(len(unique_words))

    """replace values from the arrays using frequency values of unique words from respective classes"""
    for index in range(0, len(unique_words)):
        unique_word = unique_words[index]
        if unique_word in capstone_dict.keys():
            capstone_freq[index] = capstone_dict[unique_word]
        if unique_word in thesis_dict.keys():
            thesis_freq[index] = thesis_dict[unique_word]
        if unique_word in case_dict.keys():
            case_freq[index] = case_dict[unique_word]
        if unique_word in feasibility_dict.keys():
            feasibility_freq[index] = feasibility_dict[unique_word]

    # df = pd.DataFrame({"Capstone": capstone_freq, "Thesis": thesis_freq, "Case Study": case_freq, "Feasibility Study": feasibility_freq}, index=unique_words, )

    df = pd.DataFrame(np.array([capstone_freq, thesis_freq, case_freq, feasibility_freq]), columns=unique_words, )
    df = sortByHighest(df)

    df = removeUncommonTerms(df)

    return df


def sortByHighest(data_frame):
    # something wrong with this maybe
    for term, frequency in data_frame.items():
        unsorted = frequency.to_numpy()
        descending = np.sort(unsorted)[::-1]
        data_frame[term] = data_frame[term].replace(unsorted, descending)

    return data_frame


def removeUncommonTerms(df):
    """remove words that occur less than 5 times"""
    new = {}
    for term, freq in df.items():
        if freq.sum() > 5:
            new[term] = freq
    new_df = pd.DataFrame(new)
    return new_df


class TfMonoVectorizer:
    def __init__(self, doc_label, doc_text, alpha, sqrt=False):
        self.sqrt = sqrt
        self.doc_label = doc_label
        self.doc_text = doc_text
        self.alpha = alpha
        self.terms = prepTerms(self.doc_label, self.doc_text)
        self.frequency_dist = getFD(self.terms)
        self.data_frame = getFrequencyDF(self.frequency_dist)

    def getNO(self, value_list):
        non_occ = 0
        sum_non_occ = value_list.sum()

        for i in value_list:
            non_occ += (sum_non_occ - i) / sum_non_occ
        non_occ = non_occ / (len(value_list) * 100)

        return non_occ

    def getMO(self, value_list):
        total_occ = value_list.sum()

        max_occ = value_list.to_numpy()[0] / total_occ

        return max_occ

    def getMONOGlobal(self):
        terms = []
        mono_globals = []

        for term, frequency in self.data_frame.items():
            # if total_occ > 5:
            max_occ = self.getMO(frequency)
            non_occ = self.getNO(frequency.to_numpy()[1::])
            mono_loc = max_occ * non_occ
            mono_glob = 1 + self.alpha * mono_loc
            terms.append(term)
            if mono_glob > 0:
                mono_globals.append(mono_glob)
            else:
                mono_globals.append(0)
            # else:
            #     self.data_frame.pop(term)

        return mono_globals, terms

    def getTF(self, doc_tokens, term):
        try:
            tf = doc_tokens.count(term) / len(doc_tokens)
        except ZeroDivisionError:
            tf = 0
        finally:
            if self.sqrt:
                return tf ** 0.5
            else:
                return tf

    def getTF_MONO(self):
        mono_globals, terms = self.getMONOGlobal()
        arrays = []

        for index in range(0, len(self.doc_text)):
            doc = self.doc_text[index]
            # doc = cleanText(doc)
            doc_tokens = getTokens(doc)
            mga_tf_mono = []

            for i in range(0, len(mono_globals)):
                term = terms[i]
                mono_global = mono_globals[i]
                tf = self.getTF(doc_tokens, term)
                tf_mono = tf * mono_global
                mga_tf_mono.append(tf_mono)

            arrays.append(mga_tf_mono)

        stacked_tf_mono = np.vstack(arrays)
        vector = pd.DataFrame(stacked_tf_mono, columns=terms, index=self.doc_text)
        # vector.insert(loc=0, column="Documents", value=self.doc_text)
        # vector.insert(loc=1, column="Category", value=self.doc_label)

        # vector = pd.DataFrame(stacked_tf_mono)
        return vector
        # return stacked_tf_mono


class TfEMonoVectorizer(TfMonoVectorizer):
    def __init__(self, doc_label, doc_text, alpha, sqrt=False):
        super().__init__(doc_label, doc_text, alpha, sqrt)

    def getEMO(self, value_list):
        total_occ = value_list.sum()
        value_list = value_list.to_numpy()

        max_occ = (value_list[0] + value_list[1]) / total_occ

        return max_occ

    def getEMONOGlobal(self):
        terms = []
        emono_globals = []

        for term, frequency in self.data_frame.items():
            # if total_occ > 5:
            emax_occ = self.getEMO(frequency)
            non_occ = self.getNO(frequency.to_numpy()[2::])
            emono_loc = emax_occ * non_occ
            emono_glob = 1 + self.alpha * emono_loc
            terms.append(term)
            if emono_glob > 0:
                emono_globals.append(emono_glob)
            else:
                emono_globals.append(0)
            # else:
            #     self.data_frame.pop(term)

        return emono_globals, terms

    def getTF_EMONO(self):
        emono_globals, terms = self.getEMONOGlobal()
        arrays = []

        for index in range(0, len(self.doc_text)):
            doc = self.doc_text[index]
            # doc = cleanText(doc)
            doc_tokens = getTokens(doc)
            mga_tf_emono = []

            for i in range(0, len(emono_globals)):
                term = terms[i]
                emono_global = emono_globals[i]
                tf = self.getTF(doc_tokens, term)
                tf_emono = tf * emono_global
                mga_tf_emono.append(tf_emono)

            arrays.append(mga_tf_emono)

        stacked_tf_emono = np.vstack(arrays)
        vector = pd.DataFrame(stacked_tf_emono, columns=terms, index=self.doc_text)

        return vector


class TfIgmVectorizer(TfMonoVectorizer):
    def __init__(self, doc_label, doc_text, alpha, sqrt=False):
        super().__init__(doc_label, doc_text, alpha, sqrt)

    def getIGMLocal(self, value_list):
        gravity = 0
        for index in range(len(value_list)):
            gravity += value_list[index] * index + 1

        max = value_list[0]
        igm_local = max / gravity
        return igm_local

    def getIGMGlobal(self):
        terms = []
        igm_globals = []

        for term, frequency in self.data_frame.items():
            igm_loc = self.getIGMLocal(frequency.to_numpy())
            igm_glob = 1 + self.alpha * igm_loc
            terms.append(term)
            if igm_glob > 0:
                igm_globals.append(igm_glob)
            else:
                igm_globals.append(0)
        return igm_globals, terms

    def getTF_IGM(self):
        igm_globals, terms = self.getIGMGlobal()
        arrays = []

        for index in range(0, len(self.doc_text)):
            doc = self.doc_text[index]
            doc_tokens = getTokens(doc)
            mga_tf_igm = []

            for i in range(0, len(igm_globals)):
                term = terms[i]
                igm_global = igm_globals[i]
                tf = self.getTF(doc_tokens, term)
                tf_igm = tf * igm_global
                mga_tf_igm.append(tf_igm)

            arrays.append(mga_tf_igm)

        stacked_tf_igm = np.vstack(arrays)
        vector = pd.DataFrame(stacked_tf_igm, columns=terms, index=self.doc_text)

        return vector


class LDA:
    def __init__(self, text, num_topics, num_words):
        self.num_words = num_words
        self.tokens = getTokens(text, lem=False)
        self.dictionary = corpora.Dictionary([self.tokens])
        self.corpus = [self.dictionary.doc2bow(self.tokens)]

        self.model = models.LdaModel(corpus=self.corpus,
                                     num_topics=num_topics,
                                     id2word=self.dictionary)

    def getTopics(self):
        # for i, topic in self.model.show_topics(formatted=True, num_words=self.num_words):
        return [word for word, prob in self.model.show_topic(0, topn=self.num_words)]


def getFreqTotal(frequency_distribution):
    """separate term and value from distributions"""
    capstone_dict = {item[0]: item[1] for item in frequency_distribution["Capstone"]}
    thesis_dict = {item[0]: item[1] for item in frequency_distribution["Thesis"]}
    feasibility_dict = {item[0]: item[1] for item in frequency_distribution["Feasibility Study"]}
    case_dict = {item[0]: item[1] for item in frequency_distribution["Case Study"]}

    """unify the terms from each class"""
    unique_words = reduce(np.union1d, (np.array(list(capstone_dict.keys())), np.array(list(thesis_dict.keys())),
                                       np.array(list(feasibility_dict.keys())), np.array(list(case_dict.keys()))))

    """initialize shape of the arrays"""
    capstone_freq = np.zeros(len(unique_words))
    thesis_freq = np.zeros(len(unique_words))
    case_freq = np.zeros(len(unique_words))
    feasibility_freq = np.zeros(len(unique_words))

    """replace values from the arrays using frequency values of unique words from respective classes"""
    for index in range(0, len(unique_words)):
        unique_word = unique_words[index]
        if unique_word in capstone_dict.keys():
            capstone_freq[index] = capstone_dict[unique_word]
        if unique_word in thesis_dict.keys():
            thesis_freq[index] = thesis_dict[unique_word]
        if unique_word in case_dict.keys():
            case_freq[index] = case_dict[unique_word]
        if unique_word in feasibility_dict.keys():
            feasibility_freq[index] = feasibility_dict[unique_word]

    df = pd.DataFrame({"Capstone": capstone_freq, "Thesis": thesis_freq, "Case Study": case_freq, "Feasibility Study": feasibility_freq}, index=unique_words, )
    df = removeUncommonTerms(df)
    df["Total"] = df.sum(axis=1)

    return df


def getFeatures(text, label):
    terms = prepTerms(label, text)
    fd = getFD(terms)
    df = getFreqTotal(fd)
    return df

# CSV_DIR = r"ALL_DATA(no_oversampling).csv"
# docs = pd.read_csv(CSV_DIR, index_col=False)
# background = np.array(docs["Backgrounds"])
# abstract = np.array(docs["Abstracts"])
# category = np.array(docs["Categories"])
#
#
# getFeatures("TEST", background, category)

# bg_terms = prepTerms(category, background)
# bg_fd = getFD(bg_terms)
# bg_df = getFrequencyDF(bg_fd)
# bg_df.to_csv("BG Features.csv")
#
# ab_terms = prepTerms(category, abstract)
# ab_fd = getFD(ab_terms)
# ab_df = getFrequencyDF(ab_fd)
# ab_df.to_csv("AB Features.csv")

#
# topics = LDA(text=background[300], num_topics=1, num_words=10).getTopics()
# print(topics)

# lda = LDA(background)
# lda_model = lda.createLdaModel(10)
#
# for i, topic in lda_model.show_topics(formatted=True, num_words=10):
#     print(str(i) + ": " + topic)
#     print()

# terms = prepTerms(category, background)
# fd = getFD(terms)
# df = getFrequencyDF(fd)
# df.to_csv("Hatdog.csv", index=False)

# tfmono = TfMonoVectorizer(category, background, 6)
# # monoglobal = tfmono.getMONOGlobal()
# # monoglobal.to_csv("MONO Global.csv", index=0)
# tfmono_df = tfmono.getTF_MONO()
# # print(monoglobal)
# tfmono_df.to_csv("TF MONO.csv", index=False)

# new = """HDS Bullding, 999 J.C. Aquino Avenue, Butuan City
# 
# ACLC College of Butuan City
# 
# Case Overview
# 
# On the first year, May 201 3-April 2014, the owner established standard
# procedures and protocols regarding its physical environment and customer
# service.
# 
# The KTV rooms of Lifestar Family Karaoke have noise insulation to
# reduce passage of sound from one room to another. The KTV rooms have
# two lighting options to choose from, the warm light bulb lighting or the disco
# ball lighting with three settings. Before the customers enter the KTV room, the
# employees must turn on the air conditioner and spray the room with air
# 
# freshener.
# 
# Aside from the KTV rooms, they also have a receiving area where the
# customer can order, pay, and inquire for vacant rooms. The receiving area
# has a sofa for customers to sit comfortably while waiting and is air
# conditioned. The employees also play pop music playlists on the receiving
# area. Regarding the lighting, they use white fluorescent light bulbs and mini
# LED lights in violet and blue colors. A display case of expensive and imported
# 
# liquors can also be seen.
# 
# Lifestar also has a lobby; the lobby has tables and chairs for dining and
# drinking, a mini stage for live band performances, and a big window which
# allows light to come in; by the window, is the smoking area where there are
# two chairs, a table, and an ashtray. The mini-stage has two microphones and
# a sound system. The mini-stage is decorated with Gina cloth. The lobby has
# no air conditioning. The lobby has blue LED light bulbs in each corner and
# 
# R
# SCUFI AE Ae ceirurc\)e wm RifcmMres ANMIAICTMATING 2011
# OC rl
# 
# RDO- , .
# DO-ACAD-CS16-012 RESEARCH AND DEVELOPMENT OFFICE
# HDS Building, 999 J.C. Aquino Avenue, Butuan City
# 
# d ACLC College of Butuan City
# 
# 5
# disco laser lights which emit blue, green, and red lights while the hallway has
# 
# dim lighting of blue, red, and green.
# 
# The employees must also clean the external area (street parking) and
# internal (KTV rooms, lobby, receiving area, hallway, and kitchen) before and
# after opening. Regarding the speed of service, when the customers are
# finished with their orders, the employee must prepare the drinks immediately
# and inform the kitchen staff (if they ordered food). They must also update the
# customer the time that they must wait. They must also know the product,
# price, and promotion of Lifestar to be able to respond to inquiries as well as
# suggest. They must also know how to operate all of the equipment (KTV
# machine, digital recording studio, television, sound system, microphone,
# Xbox, and air conditioner). The technician also updates the song lists every
# three months in the KTV room. In regards to responding to customer
# complaints, the employees must do these right away: listen to the customer's
# complaint, apologize, and provide a solution. Regarding the customer service
# standard procedures and protocols, the employees must greet the customer
# when they walk in and refer them as sir/ maâ€™am. For example: â€œWelcome to
# Lifestar maâ€™am.â€ Employees must also pay attention what the customer wants
# and be polite and respectful. Regarding the employeesâ€™ uniform, they are
# 
# required to wear pants, closed shoes, and a shirt provided by Lifestar.
# 
# Lifestar has a service bell in each KTV rooms. The service bell is used
# 
# by the customers to inform the employees that they need help. When the
# 
# RACUE!I Ne or Celene In RIIGINESS Annie rmatinn 2011
# rr
# 
# RDO-ACAD-C516-017 RESFARCH AND DEVFLOPMENT OFFICE
# HDS Building, 999 J.C. Aquino Avenue, Butuan City
# 
# d ACLC College of Butuan City
# 
# 6
# customer rings the bell, it automatically sends a notification of the room's
# 
# name to the receiving area; an employee will then go to that room.
# 
# The customers leave their comments, suggestions, and complaints in
# the suggestion box and Lifestar's Facebook page. On the first year, according
# to the suggestion box and Facebook page messages, the customers
# appreciated that the employees are always wearing their uniform and closed
# shoes and that they also greet them lively. They also said that the area and
# KTV rooms were clean. The customers complained that sometimes the
# employees forget to spray the air freshener after the previous customers
# 
# finished using the rooms. The sales during that year were Php 4,591,280.
# 
# On the second year, May 2014-April 2015, they renovated the lobby
# area but they still operated during the renovation. After the renovation, what
# used to be a mini stage was turned into a mini bar, an extra space for dining
# was also made above the stairs, and the open window was turned into a
# stage. The mini bar displays the different drinks that Lifestar offer, it also has
# two stools and a counter top. The stage, used for live band performances and
# open mics; has noise insulation, two microphones, two chairs, and a sound
# system. A microphone is placed on the stage during an Open Mic, when
# customers want to sing karaoke songs. Across the stage, is a LED flat screen
# television, where singers and customers can see the lyrics of the song during
# the Open Mic. Employees play music videos on the television when there are
# no live bands and open mics. The lobby still has table and chairs for dining
# 
# and drinking. The lobby is air conditioned and the smoking area is outside the
# 
# RAcurinan ar SCIeNee ts MRIICMNECeS AMAMAICTM ATI Ne 201?
# â€”â€”â€”â€”â€” IER E I SSR ee Re reee eee eee
# 
# RDO-ACAD-CS16-012 RESEARCH AND DEVELOPMENT OFFICE
# ACLC College of Butuan City
# @ HDS Building, 999 J.C. Aquino Avenue, Butuan City
# 
# 7
# entrance door of Lifestar. The smoking area has a sofa and an ashtray.
# Regarding the lighting, they use disco lights and laser lights which emit green,
# red, and blue colors. When there are only a few customers in the area, the air
# 
# conditioner is turned off and only the wall fan is turned on.
# 
# The protocol in cleaning the KTV was changed; employees must clean
# the KTV room before and after the customers enter the room and not just
# before the opening of the KTV bar. The employee turns on the air conditioner
# of the room and then sprays each corner with air freshener, two to five
# minutes before the customers enter the room. Employees are still required to
# wear pants and closed shoes but instead of shirt provided by Lifestar, they
# 
# color code their t-shirts. Example, on Monday they wear pink.
# 
# That year, the service bell stopped working which causes the customer
# to go to the receiving area when they need help. On the second year,
# according to the suggestion box and Facebook messages, the customer
# noticed a change that they started to clean the rooms before and after
# customers used it and they appreciated it. They also commented on Lifestarâ€™s
# renovation that they liked the new stage and the open mic but they still
# preferred the brightness of the light on the previous year, they also
# commented on atmosphere and said it was hot in the lobby because they will
# turn off the air condition if there were only a few customers. The employees
# were not wearing their color-coded uniform and they did not greet the
# Customers. The sales during that year were: Php 3,021,659. The sales
# 
# decreased compared from the previous year.
# 
# RACHEI AP AF ericmee im mrcmeres annemicrnatian 207]
# 
# RDO-ACAD-CS16-012 RESEARCH AND DEVELOPMENT OFFICE
# d ACLC College of Butuan City
# 
# HDS Building, 999 J.C. Aquino Avenue, Butuan City
# 
# 8
# On the third year, May 2015 - April 2016, nothing has changed with
# Lifestarâ€™s standard procedures and a protocol regarding its customer service
# 
# and physical environment and the service bell is still not working.
# 
# On the third year, according to the suggestion box and Facebook
# messages, the customer found their equipment has good performance; they
# liked the sound quality of the microphones as well as the sound system. They
# also felt that the television in the lobby is just the right size and has a nice
# resolution that makes it easy for them to see the lyrics from the stage. They
# liked that the song lists were always updated. They liked that it contains
# popular songs of different genres. They noticed that Lifestarâ€™s employees
# rarely greet them when they get in or do not even ask for repeat business.
# The lack of friendly greeting from the employees gave them the impression
# that the employees are not approachable. They said that not greeted by the
# employees can put them in a bad mood and they will feel unimportant and
# ignored. The customers also observed that the employees were not wearing
# the uniform. They also commented on the strong scents on the rooms
# because employees spray the room just right after they come in. They also
# felt that instead of getting rid of the odor of the room, the smell of the air
# freshener just mixes with the odor. They also complained about the hot
# temperature in the lobby area. The customer observed that the employees
# were not reliable with the time they promise to deliver the products; they also
# noticed that they have slow service during peak hours. For example, instead
# 
# of five (5) minutes promise time, they will wait fifteen to twenty (15-20)
# 
# RASCvWrIiAP ne ceicuesâ€™ is RIICINE SS ANMINICTRATION 201?
# rr
# 
# RDO-ACAD-CS16-012 RESEARCH AND DEVELOPMENT OFFICE
# HDS Building, 999 J.C. Aquino Avenue, Butuan City
# 
# , ACLC College of Butuan City
# 
# 9
# minutes. They also felt scared or lonely because of the dark color and dim
# 
# lights. The sales decreased to Php 2,197,951.
# 
# The sales from 2013-2016 will be shown in the figure below:
# 
# Lifestar Family Karaoke
# Comparative Annual Sales
# _ 2013 - 2016
# 
# oe __ 8 4,591,280
# ____- 3,021,659
# = 2,197,951
# 
# 4,591,280 3,021,659 2,197,951
# 
# May 2013 â€” April 2014 May 2014 â€” April 2015 May 2015 â€” April 2016
# 
# Figure 2. Comparative Annual Sales of Lifestar Family Karaoke
# 
# Pacuriap ar cricncre In ricIiMvmecc ANLAINICTRATIAN 201)
# 
# a ee ee ee ee) dt eee Bo ee ed Oe he ae on YO ed ed ed
# 
# """
# tokens = getTokens(new)
# print(lda_model[lda.dictionary.doc2bow(tokens)])
