from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import regex as re
from nltk.util import bigrams
from nltk.probability import FreqDist
from nltk.corpus import wordnet
from nltk import pos_tag
import math
import copy
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from pathlib import Path







###Tokeniser logic####


stop_words = set(stopwords.words('english'))


#only return bigrams of count more than 3
def detectBigrams(cleanedTokens):
    bigram_list = list(bigrams(cleanedTokens))
    bigram_freq = FreqDist(bigram_list)
    frequent_items = [(item, count) for item, count in bigram_freq.items() if count >3]
    joined_bigrams = [(' '.join(bigram), count) for bigram, count in frequent_items]
    return joined_bigrams


#gives all brigrams in text
def getBigrams(cleanedTokens):
    separatedBigrams = list(bigrams(cleanedTokens))
    combinedBigrams = [' '.join(x) for x in separatedBigrams]
    return combinedBigrams


#POS tag wordNet format function
def formatPOS(tag):
    if tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    

#Lemmatising function
def lemmatise(cleanedTokens):
    lemmatised = []
    taggedCleanedTokens = pos_tag(cleanedTokens)
    for wordPair in taggedCleanedTokens:
        posTag = formatPOS(wordPair[1])
        if posTag != None:
            lemmatised.append(lemmatizer.lemmatize(wordPair[0], posTag))
        else:
            lemmatised.append(wordPair[0])
    return lemmatised



#Document summarising function
def summarise(documentText):
    parser = PlaintextParser.from_string(documentText, Tokenizer("english"))
    summariser = LsaSummarizer()
    summary = summariser(parser.document, sentences_count=1)
    return summary
        

#Tokenizor
def tokenizor(text):
    soup = BeautifulSoup(text, 'html.parser')
    tags = ['title', 'h1', 'h2', 'h3', 'p']
    documentTokens = []
    unigrams= {}  #{term: {position: count}}
    bigrams = {}
    documentText = ""

    for tag in tags:
        contents = soup.find_all(tag)
        cleanedTokens = []
        for section in contents:
            cleanedContent = section.get_text()
            reS1 = re.sub("<.+?>|\n", "", str(cleanedContent))
            documentText = documentText + reS1
            tokens = word_tokenize(reS1)
            cleanedTokens += [t for t in tokens if t not in stop_words]
            cleanedTokens = lemmatise(cleanedTokens)
            tfp = tuple([(terms, cleanedTokens.count(terms)) for terms in set(cleanedTokens)])
        documentTokens.extend(cleanedTokens)
        for pair in tfp:
            if pair[0] not in unigrams:
                unigrams.update({pair[0]:{tag:pair[1]}})
            else:
                if tag in unigrams[pair[0]]:
                    temp = unigrams[pair[0]].get(tag)
                    temp = temp + pair[1]
                    unigrams[pair[0]][tag] = temp
                else:
                    unigrams[pair[0]].update({tag:pair[1]})

    

    documentSummaryStructure = summarise(documentText)
    documentSummary = ""
    for sentence in documentSummaryStructure:
        documentSummary = str(sentence)
    documentSummary = documentSummary[:100] + "..."


    
    
    bigramTupleList = detectBigrams(documentTokens)
    bigramList = []
    
    
    for pair in bigramTupleList:
        bigramList.append(pair[0])

    

    for tag in tags:
        contents = soup.find_all(tag)
        cleanedTokens = []
        for section in contents:
            cleanedContent = section.get_text()
            reS1 = re.sub("<.+?>|\n", "", str(cleanedContent))
            tokens = word_tokenize(reS1)
            cleanedTokens += [t for t in tokens if t not in stop_words]
            cleanedTokens = lemmatise(cleanedTokens)
            sectionBigrams = getBigrams(cleanedTokens)
        for item in sectionBigrams:
            if item in bigramList:
                if item not in bigrams:
                    bigrams.update({item:{tag:1}})
                else:
                    if tag in bigrams[item]:
                        temp = bigrams[item].get(tag)
                        temp = temp + 1
                        bigrams[item][tag] = temp + 1
                    else:
                        bigrams[item].update({tag:1})
            
    
    
    
    return unigrams, bigrams, documentSummary
        


                    
                
                









###Creating Inverted Index########
basePath = Path("")
filenames = [file.name for file in basePath.iterdir() if file.is_file() and file.suffix == ".html"]





            
vocabIDCounter = 0
docIDsCounter = 0

vocab = {}
docIDs = {}
invertedIndex = {}
documentSummaries = {}
bigramList = []
folderOfBooks = ""



for fname in filenames:
    path = folderOfBooks+fname
    f = open(path, "r", encoding='utf-8', errors = "ignore")
    text = f.read()
    f.close()
    


    docIDs[docIDsCounter] = fname
    tokens, bigramDict, documentSummary = tokenizor(text)
    documentSummaries[docIDsCounter] = documentSummary
    
    for x, y in bigramDict.items():
      bigramList.append(x)
      tokens.update({x:y})
    


    for t in tokens:
        word = t
        frequency = tokens[word]
        if word in vocab.keys(): 
            vocabID = vocab[word]
        else: 
            vocab[word] = int(vocabIDCounter)
            vocabID = vocabIDCounter
            vocabIDCounter += 1
        
        if vocabID in invertedIndex.keys():
            invertedIndex[vocabID].update({docIDsCounter: frequency})
            
        else:
            invertedIndex.update({vocabID:{docIDsCounter:frequency}})
            
        
    docIDsCounter += 1




#NER labels for terms
ner = {}
for term in vocab:
    process = nlp(term)
    label = process.ents[0].label_ if process.ents else None
    if label != None:
        ner.update({vocab[term]: label})








####Query Loop####
while True:

    results = {}
    query = input("What would you like to query?")
    if query == "quit": break


    queryList = word_tokenize(query)
    queryList = lemmatise(queryList)
    document_list = [] 
    vocab_list = []
    queryNER = {}

    queryBigrams = getBigrams(queryList)
    for bigram in queryBigrams:
        if bigram in bigramList:
            queryList.append(bigram)

    ###Query Expansion###
    tagged = pos_tag(queryList)

    synonyms = set()
    for wordTagPair in tagged:
        word = wordTagPair[0]
        tag = wordTagPair[1]
        formattedPOS = formatPOS(tag)
        for synset in wordnet.synsets(word, pos = formattedPOS):
            for lemma in synset.lemmas():
                if lemma.count() >= 5:
                    synonyms.add(lemma.name())

    synonyms = list(synonyms)
    queryList.extend(synonyms)

    for q in queryList:
            results[q] = set()
            if q in vocab:
                vocabID = vocab[q]
                vocab_list.append(vocabID)
                documents = invertedIndex[vocabID]
                document_list.append(documents)
                    
                for d in documents:
                    documentID = d
                    results[q].add(documentID)
                
                process = nlp(q)
                label = process.ents[0].label_ if process.ents else None
            
                if label != None:
                    queryNER.update({vocabID: label})


    candidateDocs = set()
    termDocs = {}      
    queryTermIDs = [] 

    for q in queryList:
        if q in vocab:
            vocabID = vocab[q]
            queryTermIDs.append(vocabID)
            docs = set(invertedIndex[vocabID].keys())
            termDocs[vocabID] = docs
            candidateDocs |= docs




    ###TF-IDF###
    if candidateDocs:
        ##tf
        tf = []  #[[document, frequency], [document, frequency]]

        zoneWeights = {'title':3, 'h1':2, 'h2':2, 'h3':1,'p':1}
        nerWeights = {'PERSON': 5, 'NORP': 4, 'FAC': 2, 'ORG': 5, 'GPE': 4, 'LOC': 4, 'PRODUCT': 1, 'EVENT': 1, 'WORK_OF_ART': 5, 'LAW': 1, 'LANGUAGE': 1, 'DATE': 3, 'TIME': 1, 'PERCENT': 1, 'QUANTITY': 2, 'ORDINAL': 3, 'MONEY': 3, 'CARDINAL': 5}

        queryTermIDs = list(termDocs.keys())
        docList = list(candidateDocs)

        for vocabID in queryTermIDs:
            queryTerms = []
            for docID in docList:

                if docID not in termDocs[vocabID]:
                    queryTerms.append([docID, 0])
                    continue

                tfVal = 0
                #Zone weighting
                for tag, count in invertedIndex[vocabID][docID].items():
                    tfVal += count * zoneWeights[tag]

                #NER weighting
                if vocabID in ner and vocabID in queryNER and ner[vocabID] == queryNER[vocabID]:
                    tfVal += nerWeights.get(ner[vocabID], 0)


                tfVal = 1 + math.log(tfVal, 10) #normalise tf
                queryTerms.append([docID, tfVal])

            tf.append(queryTerms)

        

        


        ###vector calculation and cosine normalisation###
        doc_freq = []
        for i in range(len(tf[0])):
            values = []
            for p in tf:
                values.append(p[i][1])
            doc_freq.append(values)

        doc_lengths = []
        for l in doc_freq:
            eucLength = math.sqrt(sum(x**2 for x in l)) #Vector calculation
            doc_lengths.append(eucLength) 

        for i in range(len(tf[0])):
            for p in tf:
                if doc_lengths[i] != 0:
                    p[i][1] /= doc_lengths[i] #Cosine similarity



        ##idf
        idf = []
        n = len(docIDs)
        for vocabID in queryTermIDs:
            df = len(termDocs[vocabID])
            idf.append(math.log(n / df, 10))
        

        
        tf_idf = copy.deepcopy(tf)
        for i in range(len(tf_idf)):
            for t_f in tf_idf[i]:
                t_f[1] = t_f[1] * idf[i]
        



        scores = [
            [tf_idf[0][i][0], sum(layer[i][1] for layer in tf_idf)]
            for i in range(len(tf_idf[0]))]



        #Sort document by score
        scores = sorted(scores, key=lambda x: x[1], reverse = True)  
        
        #Display URL and summary
        for pairs in scores[:10]:
            print("file://"+folderOfBooks+ docIDs.get(pairs[0]))
            print(documentSummaries.get(pairs[0]))
            print("")
            
        print("\n \n \n \n \n \n")

        
    else:
        print("no terms found")



