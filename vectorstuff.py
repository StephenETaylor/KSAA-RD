#!/usr/bin/python3
# coding: utf-8

"""
    A module to contain code to deal with vectors
    notice that the vectors are numpy.ndarray objects
"""
#import Bu
import sys
import numpy as np
import math

"""
    vecs and voc are the two main data structures exported by this module
    getvecs(vf,mx) initializes them from an ascii vf file.
    binloadV(vf,verbose) initializes them from a binary vector file
    binSaveV(fn) writes out a binary file from initialized data structures
    nearest(n,v) returns a list of the n words nearest vector v
    words is an inverted version of voc

"""
vecs = dict() #vectors, indexed by word
voc = dict()  #frequency rank, indexed by word
words = []    # words, indexed by position in vec array
data = None   # view of vector array by frequency rank instead of word
maxvecs = 0   # number of vectors, should be len(words)

"""
   Read vectors from vocabularyFile
"""
def getvecs(vocabularyFile, pmaxvecs):
    getvecs_unnormalized(vocabularyFile,pmaxvecs)
    normalize()

def getvecs_unnormalized(vocabularyFile,pmaxvecs):
    global vecs
    global voc
    global data
    global maxvecs
    global words
    maxvecs = pmaxvecs
    # put vocabulary in memory
    print ('extracting vocabulary from ',vocabularyFile, maxvecs)
    rpt = 1000
    voc = dict()
    vecs = dict()
    words = [None]*maxvecs
    data = np.ndarray(shape=(maxvecs,300), dtype = np.float32)
    lineNum = 0
    with open( vocabularyFile) as va:
        state = 0
        for l in va:
            v = l.split(' ')
            if state == 0:
                state = 1
                print(l)
                if int(v[1]) != 300: cry('I will not know what to do with funny size vectors')
                continue # don't update lineNum
            voc[v[0]] = lineNum
            words[lineNum] = v[0]  # simulaneously create inverse file
            #v1 = np.empty(300, dtype=np.float32) 
            v1 = data[lineNum]
            for i in range(1,301):
                v1[i-1] = float(v[i]);
            vecs[v[0]] = v1 
            lineNum += 1
            if lineNum >= maxvecs: # maybe we don't always need to wait for all?
                break
            if lineNum % rpt == 0:
                print('line',lineNum)
                if rpt < 50000:
                    rpt += rpt
                else:
                    rpt = 100000
                    
def normalize():
    global vecs
    # zero center and normalize
    count = 0
    sums = np.zeros(300, dtype= np.float32)
    for k in vecs:
        v = vecs[k]
        sums = sums + v
        count += 1
    sums = sums / count  # compute mean for each coordinate
    for k in vecs:
        v = vecs[k]
        i = voc[k]
        dout = data[i]
        #v = v - sums
        np.subtract(v,sums,out = dout)
        # normalize
        norm = math.sqrt(v.dot(v))
        #v = v / norm
        np.divide(dout,norm,out=dout)
        #vecs[k] = v 
        vecs[k] = dout

def zerocenter():
    global data, vecs
    sums = data.sum(axis=0)
    mean = sums / data.shape[0]   
    data += mean

def varnorm():
    global data
    zerocenter()
    sum2 = np.zeros(shape = (300), dtype = np.float32)
    for d in data:
        t = d * d # counting on broadcasting for t to have shape (300,)
        sum2 += t # counting on broadcasting for sum2 to have shape (300,)
    var = sum2/data.shape[0] # mean of each column is 0, so dont subtract 0**2 
    sigma = np.sqrt(var)  # and hopefully sigma also has shape (300,)
    for i,d in enumerate(data):
        data[i] = d / sigma # hoping for column-wise operations



def unitize(): # normalize all points to a radius of 1
    global data
    # as above, we do this with a loop, rather than broadcasting
    for d in data:
        factor = math.sqrt(d.dot(d))
        d = d / factor
"""
    compute list of words nearest vector w
"""

def nearest(number, w, distfunc = None): #distfunc is cos-like; for euclidean, negate
    """
        returns list of (d,neighbor) pairs, where d = distfunc(w,vecs[neighor]) 
        number is length of list to return.
        w is vector to search for
        distfunc, if present, is distance function.  it should take two 
            vectors as parameters, and return larger numbers for closer items
            vs.cosdist is appropriate if the data is unnormalized
            to use a Euclidean distance, return a negative distance, so that
             closer items have larger distance returned.
    """
    if distfunc == None:
        global vecs
        global voc
        those = [(-1,None)]*number  # set up list of number nearest objects
        for k in vecs:
            v = vecs[k]
            d = w.dot(v)    # vectors all normalized, dot product is cosine dist
            if abs(d)>those[0][0]:
                those[0] = (d,k)
                those.sort()  # maintain those as 
                            #a priority queue, most distant first
                            # identical item has cos 1;
                            # cos pi/2 = 0
                            # cos pi is furthest, -1
        return those
    else: # caller provided a cos-like distance function
        # -1e1000 is -inf, but reader won't accept -inf
        those = [(-1e1000,None)]*number  # set up list of number nearest objects
        for k in vecs:
            v = vecs[k]
            d = distfunc(w,v) 
            if abs(d)>those[0][0]: # assume closer =>( distfunc > ) # (cos-like)
                those[0] = (d,k)
                those.sort()  # looks like sort is called O(log(len(vecs)))
        return those

"""
    load vectors from a binary file
                 ordinarily:
                    the first line is original file name,
                    the second line is <vector cnt>
                    (and row-size is always 300)
                 but if the first line contains a space,
                    the line is <vector cnt> <row size>
"""
def binloadV(loadV, numvecs=None):
 # load from a binary image
        global voc,vecs,words,vsvoc,vsvecs, data, maxvecs
        with open(loadV,'rb') as fh:
            word2vec = False
            vectorfile = fh.readline().decode('utf8').strip()
            if vectorfile.find(' ') == -1:
                filemaxv = maxvecs =  int(fh.readline().decode('utf8').strip())
            else:
                word2vec = True
                line = vectorfile.split()
                maxvecs = int(line[0])
                if int(line[1]) != 300:
                    sys.stderr.write('?funny row width: ')
                    sys.stderr.write(line[1])
                    sys.stderr.write('?for file:')
                    sys.stderr.write(loadV)
                    # noisy message is a warning.  But just continue?
                # for word2vec format, binary and ascii interleaved
                # so allocate binary now
                vsbase = np.ndarray(shape = (maxvecs,300), dtype= np.float32)

            if type(numvecs) != type(None) and numvecs < maxvecs:
                maxvecs = numvecs
            words = [None]*maxvecs
            for i in range(maxvecs):
                if word2vec == False:
                    words[i] = fh.readline().decode('utf8').strip('\n')
                else: # word2vec format delimits with space
                    x = b'' # string of bytes
                    y = fh.read(1) # returns a byte
                    while y != b' ':
                        x += y
                        y = fh.read(1) # fh is buffered, nosyscall. still slow
                    words[i] = str(x,'utf8')
                    vsbase[i] = np.fromfile(fh,dtype=np.float32,count=300)
            if not word2vec:
                vsbase = np.fromfile(fh,dtype=np.float32,count = 300*maxvecs)
                vsbase = vsbase.reshape((maxvecs,300))
            vsvecs = dict()
            vsvoc = dict()
            for i in range(maxvecs):
                vsvecs[words[i]] = vsbase[i]
                vsvoc[words[i]] = i
            voc = vsvoc
            vecs = vsvecs
            data = vsbase
            for w,i in voc.items():
                words[i] = w
            

"""
    save vectors into a binary file
"""
def binSaveV(vectorfile,verbose=True):
# save vector file to vectorfilename+'.bin'

    global maxvecs, words, vecs # true by default, since none are modified
    if verbose: print('saving vector file')
    with open(vectorfile+'.bin','wb') as of:
        of.write(vectorfile.encode( encoding = 'utf-8'))
        of.write(b'\n')
        of.write(str(maxvecs).encode( encoding = 'utf-8'))
        of.write(b'\n')
        for i in range(maxvecs): # save words array. reconstitute voc onload
            of.write(words[i].encode( encoding = 'utf-8'))
            of.write(b'\n')
        for i in range(maxvecs): # save vectors
            vecs[words[i]].tofile(of)


def cosdist(v, w):
# take cosine difference between two possibly unnormalized vectors
# if vectors known to be unit normalized, use naked dot product for performance
    num = v .dot  (w)
    normv2 = v .dot (v)
    normw2 = w .dot (w)
    return num / math.sqrt(normv2 * normw2)


def xform(Cmatrix):
    """
        transform the vector data by multiplying by the Cmatrix
    """
    global data
    data = data @ Cmatrix
    for i in range(maxvecs):
        vecs[words[i]] = data[i]


"""
    This is a porting of the Zform class from java to python
"""

#public static int debugCount = 10;
"""
 * adjust transforms a string 
 * from regular UTF8 Arabic, to a more ambiguous form which Zahran hoped
 * would avoid having some misspelled versions of the word in the embedding.
 * In particular, ا أ إ and آ are all replaced by ا ; the character ة  is
 * replaced by ه ; and a ي  at the end of a word is replaced by ى .
 * Since the other spellings do not occur in Zahran's embedding, 
 * not doing this transformation would incur multiple out-of-vocabulary
 * problems.
 * It seems likely that with another embedding, we might omit these
 * substitutions.
 * In addition, we also remove 
 *     fatha, kasra, damma, (the short vowels a, i, u)
 *     fatatan, kasratan, dammatan (nunnation markers)
 *     sukkun (no vowel after consonant)
 *     shadda (consonant doubling)
 *     madda (hamza riding alif is followed by long vowel)
 *     tatweel  (justification character)
 * 
 """
# this static array is used instead of allocating a StringBuffer
#static final int tt = 33;
#static char[] buffer = new char[tt];
#static int bufferCapacity = tt;

"""
 called with an arabic word in utf8 string
 returns a word modified as described above
"""
def adjust(aword):
    #if (aword.length() > bufferCapacity){
    #        int nlen = (bufferCapacity + 16) & 0xfffffff0;
    #        buffer = new char[nlen];
    #        bufferCapacity = nlen;
    #}
    buffer = []
    #int firstFree  = 0;
    for i,c in enumerate(aword): #(int i = 0; i< aword.length(); i++):
        #c = ord(ch)               # char c = aword.charAt(i);
        if ( c == '\u0623' ) : # hamza riding on alif
            buffer.append( '\u0627' ) # alif no hamza
        elif ( c == '\u0622' ) : # hamza with long vowel on alif
            buffer.append( '\u0627' )
        elif ( c == '\u0625' ) : # hamza under alif
            buffer.append( '\u0627' )
        elif ( c == '\u0629') : # taa marbuta
            buffer.append( '\u0647'  )       # heh
        elif ( c == '\u064A' ):  # yaa
            if (i == len(aword)-1) : #was aword.length()
                    buffer.append( '\u0649') # alif maqsura
            else: buffer.append( '\u064A') #yaa other than at end of word
        elif (c >= '\u064B' and '\u0653' >= c): # c one of fathatan,
            continue # dammatan, kasratan, fatha, damma, kasra, shadda, 
                              # sukkun, madda  ... ignore
        else: buffer.append( c )
    
            
#if (debugCount-- > 0){
#        System.out.println("adjusting "+aword+" to "+answer);
#    }
    return ''.join(buffer)



if __name__ == '__main__':
    getvecs('synsp/wiki.en.vec', 30000)

    binloadV('20170920.vec.bin')
    for i in range(100):
        x = nearest(10,vecs['في'])
    print (x)
