"""
  This program reads in the Cross-lingualReverseDictionary
  json files
"""
import gensim.models as gm
import json
import numpy as np
import numpy.linalg as nl
import pickle
import random
import sys
import time
import vectorstuff as vs

Rank_Cache = True # a feature introduced post-competition
Multi_File = False # a feature introduced post-competition, but not yet coded

English_stopwords = 'and for of the with on or in a over under through by to'.split()
English_stopwords += 'before after during against without around since from'.split()
English_stopwords += 'him her his hers he she it someone something they them theirs'.split()
English_stopwords += ', ; . ? !'.split()
English_stopwords = set(English_stopwords)

start = time.time()
def check_time(label):
    now = time.time()
    interval = now-start
    print(label, "%5.3f"%interval)

class mymodel:
    """
    presumably there will be more methods to be added later...
    This class gathers the data from a dictionary into some more
    useful/efficient structures

    These include:
        words  a dict, indexed by head-word, which returns 
            a list of ids corresponding to the word (possibly only one)
            the indices all occur in the training data and correspond
            to word senses, so a single word may have a lot of senses
            The head-words are by word-sense, and correspond to a unique
            id, but not a unique spelling.
        swords a dict, available if model built with si_flag.  The index 
            is a simplified form of the word, returning a list of words
            each of which can be used to get a list of ids.
            The justification for swords is the ambiguity of arabic orthograpy.
            Many words are homographs , especially when unvocalized.
            (a word may be spelled with or without vowels.)
            (In addition some verb headwords include associated prepositions...)
            (And since Arabic verbs are inflected, it would make sense to
             include a way to quickly find any verbs a string in the gloss
             might be. We count on the "likeness search" to end up ignoring
             unlikely senses, including declensions of off-the wall verbs.)
        lwords a list of words from the definition. Indexed in parallel
            (that is, with the same offset) as lids, enids, arword, arpos...)
        lids (list (parallel offset)) of ids.  There is also an ids dict, see below.
        ids a dict providing an offset into lids, lwrods, enids, arword, etc.
        enids (list (parallel offset))(available if model built with enid_flag) id of an english 
            match
        arwords (list (parallel offset))(available if model built ith ae_flag) the arabic word
        arpos (list (parallel offset))(available if model built ith ae_flag) the arabic part of 
            speech
        glosses (list (parallel offset))) the gloss of the word)
        npvecs (list (parallel offset)) sgns numpy vector for the word
        npvece (list (parallel offset)) electra numpy vector for the word
        ranks (list (parallel offset))  sgns rank hints
        ranke (list (parallel offset))  Electra rank hints


    """

    def __init__(self,dataset, enid_flag=False, ae_flag=False, si_flag=False, se_flag=False):
        """
        initialize the model from a list of dicts as returned by a call
        to read_json_file
        enid_flag == True : entries should include enid
        ae_flag == True : entries have 'ar'+X and X entries for word, pos, gloss
        si_flag == True : make entries in the sword dict 

        """

        self.norms = None

        words = dict() # a dict of strings and the associated ids
        lwords = []
        lids = []
        if enid_flag:
            self.enids = []
        if ae_flag:
            self.arwords = []
            self.arposes = []
            self.arglosses = []
        #if si_flag or se_flag:
            #self.swords = dict()
        self.swords = dict()

        ids = dict() # a dict of ids with associated offsets
        vectors = []  # list of skipgram vectors, indexed by offset
        vectore = []  # list of electra vectors, indexed by offset
        glosses = []
        next_offset = 0

        dups = dict()
        for wdic in dataset: # for each word in the list
            idval = wdic['id']
            lids.append(idval)
            if enid_flag:
                self.enids.append(wdic['enId'])
            if ae_flag:
                self.arwords.append(wdic['arword'])
                self.arposes.append(wdic['arpos'])
                self.arglosses.append(wdic['argloss'])
            wordval = wdic['word']
            lwords.append(wordval)
            if si_flag:
                # although some of the same arguments apply to english words,
                # the code here only works for arabic words.  
                # although the task 2 training data pushes the arabic data
                # into arword, arpos, argloss, the task 1 data 
                # names are word, pos, gloss for arabic data (and there is no
                # task 1 english data )
                siword = vs.adjust(wordval)
                if siword in self.swords:
                    self.swords[siword].append(wordval)
                else:
                    self.swords[siword] = [wordval]
                lsi = siword.strip().split()
                if len(lsi) > 1:
                    fword = lsi[0]
                    l = self.swords.get(fword,None)
                    if l is None: self.swords[fword] = l = []
                    l.append(wordval)
                # heres where the code to inflect siword (which stands for
                #  simplified or inflected word) would go.
                if wdic['pos'] == 'V': #a verb
                    if len(lsi) == 1:
                        fword = wordval
                    # I know that the rule is more complicated...
                    for iword in ['ا'+fword, 'ت'+fword, 'ن'+fword, 'ي'+fword, 'ي'+fword+'ون' ]:
                        if iword in self.swords:
                            self.swords[iword].append(wordval)
                        else: self.swords[iword] = [wordval]
                elif wdic['pos'] == 'noun':
                    # masculine singular nouns are inflected for accusative case
                    # but this wordform is usually an adverb with no obvious
                    # connection to the meaning of the root...
                    # again, the likeness comparison might handle this.
                    pass
                 
            elif se_flag: # put 'adjusted' and inflected words for English
                          # into swords.  
                seword = wordval.lower()
                self.add_sword(seword,wordval)

                # check to see if the 'word' is a phrase.
                # if so, try to put the core into sword
                temp = seword.strip().split()
                if len(temp) > 1:
                    y = []
                    for x in temp:
                        if x in English_stopwords:
                            temp.delete(x)
                    if len(temp) == 1:
                        self.add_sword(temp[0],wordval)


                #check the POS, and add inflections
                if wdic['pos'] == 'noun':
                    if len(temp) == 1:
                        self.add_sword( temp[0]+'s',wordval)
                elif wdic['pos'] == 'verb':
                    if len(temp) == 1:
                        for suffix in ['s','ed','ing']:
                            self.add_sword(temp[0]+suffix, wordval)


            glossval = wdic['gloss']
            glosses.append(glossval)
            if len(glossval.strip().split()) == 1:
                self.add_sword(glossval.lower(), wordval)
            # consider checking for comma-separated list of glosswords...

            
            if wordval in words:
                words[wordval].append(idval)
            else:
                words[wordval] = [idval]

            if idval in ids:
                # if we get this unexpected error, the next_offset will break
                #print('duplicate sense definition', idval, enids[-1] if ae_flag else '')
                prev_offset = ids[idval]
                #if lwords[prev_offset] != wordval:
                #    print('Both',lwords[prev_offset], 'and' ,wordval, 'have id',idval)
                    
                if idval in dups:
                    dups[idval] += 1
                else:
                    dups[idval] = 1

            ids[idval] = next_offset

            vector = []
            for col in wdic['sgns']:
                vector.append(float(col))
            vectors.append( vector )

            vector = []
            for col in wdic['electra']:
                vector.append(float(col))
            vectore.append( vector )

            next_offset = len(vectors)
            
        npvecs = np.array(vectors, dtype=np.float32)
        npvece = np.array(vectore, dtype=np.float32)

        (self.words,self.ids, self.lwords,  self.lids, self.glosses, 
                self.npvecs, self.npvece) = (
                    words,ids, lwords,  lids, glosses, npvecs, npvece)

        if len(dups) >0:
            duplicates = list(dups.values())
            duplicates . sort(reverse = True)
            print(len(dups), 'duplicates',
                    duplicates[0], 'max', 
                    sum(duplicates),'total')

        #return (words,ids, lwords,  glosses, npvecs, npvece)

    def add_sword(self, key, wordval):
        if key in self.swords:
            self.swords[key].append(wordval)
        else: self.swords[key] = [wordval]

    def add_ranks(self,N):
        self.ranks = self.add_rank( self.npvecs,N)

    def add_ranke(self,N):
        self.ranke = self.add_rank( self.npvece,N)

    def add_rank(self,  npvec, N):
        """
        npvec is one of self.npvecs or self.npvece.  
        depending on the input file, these might have 45500 or 65000
        vectors. 
        So the npvec.shape is (length,width) where width is one of 300 or 256
        The plan is to create an array rank.shape=(length,N) of cosine values, 
        rank[i,j] is the cosine value such that there are length*(j+1)/N
        vectors v in npvec such that rank[i,j] >= normalize(npvec[i].dot(v)

        There are a couple of possible implementations, but it seems likely we
        want to do the computation {m rows of npvec} (m x width) @ npvec.T
        to get answer (m x length), where m is a value which will conveniently
        fit in memory.  Since the npvecs arrays are float32, the answer array 
        takes up 4*m*length bytes.  (Since npvec takes up something like 1200
        bytes / vector, a value for m ~ 1000 gvies an answer array taking 3-4
        times the space occupied by the vectors.

        Then sorting the rows of answer gets the sorted list of cosines for
        the m rows, from which we can copy to rank the cosines at j/N;

        repeating this process ceiling(length/width) builds the whole rank array
        """
        length,width = npvec.shape
        norms = nl.norm(npvec, axis = 1)
        npvec = npvec / norms[:,np.newaxis]     # a normalized copy of the relevent npvec*
        rank=np.ndarray((length,N), dtype=np.float32)
        m = 1000
        for moffset in range(0,length+m,m):
            check_time('ranks at '+str(moffset))
            lastm = min(length,moffset+m)-1
            answer = npvec[moffset:moffset+m,:] @ npvec.T
            for i,mi in zip(range(moffset),range(moffset,lastm)):               
                sansmi = np.sort(answer[i,:])
                for j in range(0,N):
                    sindex = (j)*length//N
                    rank[mi,j] = sansmi[sindex]
        return rank

    def pickle(self, filename):
        with open(filename,'wb') as fob:
            pickle.dump(self, fob)

    def unpickle(filename): # this is a class function; called without self
        with open(filename,'rb') as fib:
            retval = pickle.load(fib)
        return retval
            
    def rank_similarity_ide(self,id1,id2):
        """
            returns percentile of similarity of two ids.
            a value of say, 0.4 means that 60% of the vectors
            in npvece are closer in cosine distance to id1
            than id2 is.
            This is not necessarily symmetric.  There exist 'hubs'
            which are near neighbors to many ids, so one of those neighbors
            has not the same rank-similarity in the hub,id direction
            as in the id,hub direction.  (check maxlis code for assumptions
            of symmetry...)
        """
        N = self.ranke.shape[1]
        offset1 = self.ids[id1]
        offset2 = self.ids[id2]
        v1 = self.npvece[offset1]
        n1 = nl.norm(v1)
        v2 = self.npvece[offset2]
        n2 = nl.norm(v2)
        cos = v1 @ v2 / n1 /n2
        r1 = mymodel.interpolate(cos,self.ranke,offset1,N)
        r2 = mymodel.interpolate(cos,self.ranke,offset2,N)
        # compute  mean of both-way estimates, and return a 'distance'
        return 1-(r1+r2)/2

    def interpolate(cos,rank,offset,N): # a class function
        # interpolate
        j1 = np.searchsorted(rank[offset], cos, side = 'left')
        if j1 == 0: return 0 # j1< rank[0]  , but 0 is limit-value for rank
        elif j1 == N: # extrapolate from top cached value
                      # because slope rises sharply between (N-1) and (N)
                      # linear extrapolation gives a value which is too high...
            hival = 1
        else: #normal case
            hival = rank[offset,j1]
        loval = rank[offset,j1-1]
        cosdif = hival - loval
        return j1 - 1 + (cos - loval)/cosdif

            
    def rank_similarity_ids(self,id1,id2):
        N = self.ranks.shape[1]
        offset1 = self.ids[id1]
        offset2 = self.ids[id2]
        v1 = self.npvecs[offset1]
        n1 = nl.norm(v1)
        v2 = self.npvecs[offset2]
        n2 = nl.norm(v2)
        cos = v1 @ v2 / n1 /n2
        r1 = mymodel.interpolate(cos,self.ranks, offset1, N)
        r2 = mymodel.interpolate(cos,self.ranks, offset2, N)

        # compute  mean of both-way estimates, and return a 'distance'
        return 1-(r1+r2)/2
        

    def get_svec(self,id):
        """
        return sgns vector for a sense-id
        """
        offset = self.ids[id]
        return self.vectors[offset]

    def nearbys(vector, number):
        """
        find senses-ids nearby in sgns space for a vector
        """

        if self.norms is None:
            self.norms = nl.norm(vectors)

        nvector = vector/nl.norm(vector)

        coses = vectors @ nvector
        coses = coses / self.norms

        idx = np.argsort(coses)
        answer = []
        for i in idx[-number:]:
            answer.append(lids[i])

        return answer


    def get_evec(id):
        """
        return electra vector for a sense-id
        """
        offset = self.ids[id]
        return self.vectore[offset]

    def nearbye(vector, number):
        """
        find sense-ids nearby in electra space for a vector
        """

        if self.norme is None:
            self.norme = nl.norm(vectore)

        nvector = vector/nl.norm(vector)

        coses = vectore @ nvector
        coses = coses / self.norme

        idx = np.argsort(coses)
        answer = []
        for i in idx[-number:]:
            answer.append(lids[i])

        return answer



mine = '/home/staylor/summer23/ardictionary/' #mycode/'
dirtrain = mine+'Cross-lingualReverseDictionary/'
dirtrain2 = mine+'Cross-lingualReverseDictionary/'
dirtrain1 = mine+'ArabicReverseDictionary/'
englishdict = dirtrain + 'English Dictonary/' + 'en.complete.with.id.json'

def read_json_file(PATH_TO_DATASET):
    with open(PATH_TO_DATASET, "r") as file_handler:
        dataset = json.load(file_handler)
    return dataset

def read_xml_file(PATH_TO_DATASET):
    pass
    return None




def main():

    train = read_json_file(dirtrain+'ar.en.train.json')
    check_time('read_train')
    bigdic = read_json_file(mine+'ArabicReverseDictionary/'+'ar.train.json')
    check_time('read_bigdic')
    print ('bigdic entries:', len(bigdic))


    dev = read_json_file(dirtrain+'ar.en.dev.json')
    check_time('read_dev')
    eng = read_json_file(englishdict)
    check_time('read_eng')
    model = mymodel(eng)
    check_time('builtmod')
    model.pickle('model.pickle')
    modela = mymodel(bigdic, si_flag=True)
    check_time('builtmoda')
    modela.pickle('modela.pickle')

def main2a():
    # adding new sword choices for english
    eng = read_json_file(englishdict)
    model = mymodel(eng, se_flag = True)
    model.pickle('model.pickle')


def main2():
    modela = mymodel.unpickle('modela.pickle')
    modela.add_ranks(100)
    modela.add_ranke(100)
    check_time('marar')
    modela.pickle('marar.pickle')
    check_time('marar pickle')
    pass

def main3():
    model = mymodel.unpickle('model.pickle')
    model.add_ranks(100)
    model.add_ranke(100)
    check_time('mengr')
    model.pickle('mengr.pickle')
    check_time('mengr pickle')
    pass


def main4():
    """
    using the rank data generated in main2 and main3, scatterplot cos at the
    100 percentiles.
    """
    for lang in 'ara','eng':
        mod = mymodel.unpickle('m'+lang+'r.pickle')
        for lab,vec,rank in [('sgns',mod.npvecs,mod.ranks),
                            ('Electra', mod.npvece,mod.ranke)]:
            with open(lang+'-'+lab+'.dat','w') as fo:

                length,width = rank.shape
                for _ in range(100): # sample a few random spots
                    while True:
                        i = random.randint(100,length-100)
                        if rank[i,12] == 0: continue
                        if vec[i,0] != 0: break #discard zero vectors:w


                    for pct in range(100):
                        c = rank[i,pct]
                        print(pct, c, i, mod.lids[i], file=fo)

def main5():
    main2()
    main3()

def main6():
    arate = read_json_file(dirtrain2 + 'ar.en.train.json')
    marate = mymodel(arate, enid_flag=True, ae_flag = True)  
    marate.pickle('marate.pickle')  # not using rank on this model
    main4()

def main7():
    main()
    main2()
    main3()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        main7()
    elif sys.argv[1] == '0':
        main()
    elif sys.argv[1] == '1':
        main1()
    elif sys.argv[1] == '2':
        main2()
    elif sys.argv[1] == '2a':
        main2a()
    elif sys.argv[1] == '3':
        main3()
    elif sys.argv[1] == '4':
        main4()
    elif sys.argv[1] == '5':
        main5()
    elif sys.argv[1] == '6':
        main6()
    elif sys.argv[1] == '76':
        main7()
        main6()
