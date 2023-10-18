#!/usr/bin/env python3
"""
This program is intended 
to read a test file, in json format,
and write a results file, also in json format
    for each entry in the test file, which will include 
        an English gloss

        build a vector from the words in the gloss
        and write a corresponding result which will include the
             input data from the test file, and also
             an 'id' and 'word' entry corresponding to the arabic word defined 
             by the gloss.


my proposed strategy:  
    guess the senses of each word in the gloss,
    and sum the corresponding vectors, to get 'Result'.  
    Find the closest vector to 'Result' in the arabic space, and return it.  

Problems with this strategy:
    0) unknown how productive it would be.
    1) The ardictionary.zip can't be expanded, so I can't get a broad Arabic
       space.  Online arabic embeddings aren't sense coordinated so I might
       get a word, but not an id.
    2) The english and arabic vectors for the same sense are not the same,
       so I need a cross-lingual transform.   Luckily the train set
       includes >2k senses, so I can probably build that transform.



"""
#import itertools as it
import math
import numpy as np
import numpy.linalg as nl
import sys
import random
import readCLRD as rC
import vectorstuff as vs

#global parameters
task=1  #either 1 or 2
electra = True
pickleModels = True
rank_distance = False
mymodel = rC.mymodel

if len(sys.argv)>1: task = int(sys.argv[1])
if len(sys.argv)>2 and (sys.argv[2] == 'S' or sys.argv[2] == 's'): electra = False

def main():
    global mara, meng
    if pickleModels:
        mara = rC.mymodel.unpickle('marar.pickle')
        meng = rC.mymodel.unpickle('mengr.pickle')
        marate = rC.mymodel.unpickle('marate.pickle')
        rC.check_time('unpickled models')
    else:
        ara = rC.read_json_file(rC.mine + 'ArabicReverseDictionary/' + 'ar.train.json')
        eng = rC.read_json_file(rC.englishdict)
        arate = rC.read_json_file(rC.dirtrain2 + 'ar.en.train.json')

        rC.check_time('read jsons')
        mara = rC.mymodel(ara,si_flag=True) # has enId, but are all '[]'
        meng = rC.mymodel(eng) #,ae_flag=True,enid_flag=True)  #eng lacks arword argloss, enId, etc.
        marate=rC.mymodel(arate, enid_flag=True, ae_flag=True)

        rC.check_time('built models')

    # now build X and Y vectors such that X[a] is the arabic embedding
    #  and Y[a] is the english embedding for some sense a.
    count = max(len(marate.ids),5000)

    if electra:
        cva = marate.npvece.shape[1]
        cve = meng.npvece.shape[1]
    else:
        cva = marate.npvecs.shape[1]
        cve = meng.npvecs.shape[1]
    X = np.zeros((count,cva), dtype=np.float32)
    Y = np.zeros((count,cve), dtype=np.float32) # meng sgns vectors only 256
    for i,(idv,ioff) in enumerate(marate.ids.items()):
        en_id = marate.enids[ioff]
        yoffset = meng.ids[en_id]
        if electra:
            X[i] = marate.npvece[ioff]
            Y[i] = meng.npvece[yoffset]
        else:
            X[i] = marate.npvecs[ioff]
            Y[i] = meng.npvecs[yoffset]
        count += -1
        if count <= 0: break
    # now we want to find M: X @ M = Y
    # and N: X = Y @ N
    # so we compute the pseudo inverse of X and Y
    M = nl.pinv(X) @ Y
    N = nl.pinv(Y) @ X
    rC.check_time('built Xform')

    # a little sanity check on the data
    for m in meng, mara, marate:
        wordphrases = 0
        glosswords = 0
        sensecnt = 0
        for w,g in zip(m.lwords, m.glosses):
            wi = m.words[w]
            if len(wi) > 1: 
                sensecnt += 1
                #print(w,'has senses',wi) # check for multiple senses
            if len(w.strip().split()) > 1: wordphrases += 1
            glosswords += len(g.strip().split())
        print('word phrases frac', wordphrases/len(m.words))
        print('avg gloss',glosswords/len(m.glosses))
        print('senses/word',sensecnt/len(m.lwords))


        

    # consider the task2 test file:
    if task == 1:
        tmod = mara
        tmod_npvec = tmod.npvece if electra else tmod.npvecs
        testfile = rC.dirtrain1+'ar.test.json'
    elif task == 2:
        tmod = meng
        tmod_npvec = tmod.npvece if electra else tmod.npvecs
        testfile = rC.dirtrain2+'ar.en.test.json'
    else: complain('about that value for task:',task)

    if electra:
        electra_sgns = 'electra'
    else:
        electra_sgns = 'sgns'
    
    # some statistics to collect
    sword_used = 0
    total_test_words = 0
    total_swords = 0
    total_glosswords_missing = 0
    total_gloss_words = 0
    total_one_glosses = 0
    total_zero_glosses = 0
    total_maxvals = 0

    oflag = 'e' if electra else 's'
    with open('output'+str(task)+oflag+'.json','w') as fo:
        print('[\n', file=fo)
        dev2 = rC.read_json_file(testfile)
        first_in_file = True
        for d in dev2:
            print(d.keys())
            print(d['gloss'])
            gloss_words = d['gloss'].strip().split()
            gloss_word_missing = 0
            glossi = []  # a list of possible ids and their vectors for the gloss
            for w in gloss_words:
                wa = vs.adjust(w)
                if  w in tmod.words:
                    glossi.append(tmod.words[w])
                #elif task == 1 and  wa in tmod.swords:
                elif   wa in tmod.swords:
                    li = []
                    for s in tmod.swords[wa]:
                        li += tmod.words[s]
                    sword_used+= 1
                    glossi.append(li)
                else:
                    gloss_word_missing += 1
                    continue
                # could also discard some stop words here...

                # build list of senses and their vectors
                #for wsense in tmod.words[w]:
                    #li .append((wsense, nvs))
            print(gloss_word_missing, 'missing dictionary entries', len(glossi), 'words processed in gloss')
            if len(glossi) > 0:
                mli, contrib = maxids(glossi) #choose most compatible senses
                maxval = sum(contrib)
                mmax = max(contrib)
            else:
                mli = contrib = []
            if len(mli) < 3:
                mask = [1.0] * len(mli)
            elif math.isclose(0,mmax) or math.isclose(0,maxval):
                mask =  [1.0] * len(mli)
            else:
                if rank_distance:
                    mask = [x/mmax for x in contrib]
                else:
                    mask = [math.sin(x* math.pi/2 /mmax) for x in contrib]

            vecs = np.zeros(tmod_npvec.shape[1], dtype=np.float32)
            for wsense, msk in zip(mli,mask):
                vsense = tmod_npvec[tmod.ids[wsense]] 
                nvs = vsense/nl.norm(vsense)  # normalize each vector just once
                vecs += nvs * msk
            if task == 2:
                vecs = vecs @ N # transform to arabic space
                
            if electra_sgns in d: # development, not test...
                dsgns = np.array(eval(d[electra_sgns]))
                dsgns = dsgns / nl.norm(dsgns)
                vecs = vecs /nl.norm(vecs)
                dang = angle(dsgns, vecs)
                print (d['id'], dang)
            
            # write json file

            if first_in_file:
                first_in_file = False
            else:
                print(end=',',file=fo)
            print ('{ "id": "'+str(d['id']+'",'),
                    '"gloss" : "'+escape_quotes(d['gloss'].strip())+ '",\n',
                  ' "'+electra_sgns+'" : ', strv(vecs),'}', file =fo)     

            rC.check_time('id '+str(d['id']))
            
            # gather some statistics to print at the end
            total_test_words += 1
            total_swords += sword_used
            total_glosswords_missing += gloss_word_missing
            total_gloss_words += len(glossi)  #these are total used in search
            total_one_glosses += 1*(len(glossi) == 1)
            total_zero_glosses += 1*(len(glossi) == 0)
            total_maxvals += maxval
        print(']', file=fo)

    print('tests', total_test_words, 
            'total_swords' , total_swords,
            'total_glosswords_missing', total_glosswords_missing,
            'total_glosswords', total_gloss_words,
            'total_one_glosses', total_zero_glosses,
            'total_zero_glosses', total_zero_glosses,
            'mean maxval', total_maxvals/total_test_words)



def strv(vec):
    """
    takes an np vector, and returns a string showing it as a python list of floats
    """
    vlength = vec.shape[0]
    numbers = []
    for i in range(vlength):
        numbers.append('%1.6e,'%vec[i])
    numbers[-1] = numbers[-1][:-1]  # discard last comma
    lines = []
    six = 1
    for i in range(0, vlength+six, six):
        lines.append(' '.join(numbers[i:min(i+six,vlength)]))
    answer = '[' + ('\n'.join(lines)) + ']'
    return answer

def escape_quotes(stri):
    if chr(34) not in stri: return stri
    answer = ''
    for ch in stri:
        if ch == chr(34):
            answer += '\\' + ch
        else: answer += ch
    return answer
        


def maxids0(lis):
    """
    This function is supposed to choose the list of senseids which minimizes 
    the total angles between the possible id vectors.
    It uses angles because summing cosines is inappropriate, cosines and
    cosine distances don't have the same triangle relation as angles.

    This is a simple reference implementation.  However, it is probably not the
    most efficient.  The 'angle()' call is pretty expensive, and this version
    does P*(n-1)*n/2 of them every time it is invoked; where P is number of 
    different lists returned by the genlis generator, which is the product of 
    the length of the elements of lis; and n is the number of elements of lis.

    It seems likely that genlis could be modified to return the
    sum of the angles, so that we could  trade off caching the cos values for
    recomputing only the angles for those elements which have changed since
    the last iteration.
    """
    minsofar = None
    angleCache = dict()
    for ls in genlis(lis):
        # compute sum of angles between all pairs
        x = 0
        for i in range(len(ls)-1):
            for j in range(i+1, len(ls)):
                x += angle_ids(ls[i],ls[j], angleCache)
        if minsofar is None or minsofar < x:
            minsofar = x
            bestsofar = ls
    return bestsofar, minsofar


def genlis(lis):
    """
    recursive generator to return cartesian product of list elements
    """
    if len(lis) == 0:
        yield None
    elif len(lis) == 1:
        for senseid in lis[0]:
            yield [senseid]
    else:
        for senseid in lis[0]:
            i0 = [senseid]
            for l in genlis(lis[1:]):
                yield i0 + l

def maxids2(lis):
    """
       A probalistic search
       for Most Likely Sense-ids
       assumption is that the most compatible senses are the correct ones for
       the gloss.
       Here we randomly pick many times a set of senses, and then search
       nearby sets for a local minimum.
       return the best of these as an approximation to the best set of ids

       each list of senses can also be thought of as a graph, with weighted
       edges between each sense.

       the difference between ids is an angle; an angle of zero (cos=1) is
       great, but an angle of pi/4 (~.7825, cos ~ .707) or pi/3 (~1.05 cos .5)
       is pretty good also, while an angle of pi/2 (1.507, cos 0) is terrible.

    """
    if len(lis) == 0:
        return [],[]
    cache = dict()
    bestsofar = None   # a final return value.  bestrial produced it.
    prod = 1
    for l in lis:
        prod = prod*len(l)
    for _ in range(min(prod,100)):  # aim for a set composition in the 99TH Percentile
        trial = []
        # build random trial list of senses
        for idvs in lis:
            trial.append(idvs[random.randint(1,len(idvs))-1])
        # now score this set
        #  cache all the current distances for trial in dist,
        # which is a upper-right triangular matrix (because symmetrical)
        # with the diagonal zero (because distance from trial[i] to trial[i]
        # is zero for all i.)
        dist = [None]*len(trial)
        dist[-1] = []
        for i in range(len(trial)-1):
            dist[i] = [angle_ids(trial[i],trial[i+j],cache) 
                                for j in range(1,len(trial)-i)]
        # now compute, using the dist array, the contribution of each of the
        # items trial[i] to the sum of all the connections between nodes in
        # the complete graph
        contribution = []
        for i in range(len(trial)):
            contribution.append(sum(dist[i]) +
                                sum([dist[c][i-c-1] for c in range(i)]))
        score = sum(contribution)/2  # every link has two ends...

        # look for a possible change
        pchgCol = pchgNewId = pchgOldId = pchgChange = None
        while True:  # until we find a local minimum
            # examine each column of trial, and
            for col in range(len(lis)):
                oldId = trial[col]

                #compute contribution of current occupant of column to score
                oldDiff = contribution[col]

                # now consider ids allowed for this column
                for newId in lis[col]:
                    #compute newId's contribution to the new score
                    newDiff = 0
                    newDists = []
                    for i in range(len(lis)):
                        if i == col:
                            newDists.append(0)
                            continue
                        temp = angle_ids(newId,trial[i],cache)
                        newDists.append(temp)
                        newDiff += temp

                    # does this change make score smaller?
                    if newDiff >= oldDiff: continue # would be worse
                    change = newDiff - oldDiff # so change is negative
                    if change < 1e-7 : continue # likely float32 complication
                    if pchgCol is None or  change < pchgChange:
                        # save the bestpotential change until all are examined
                        pchgCol = col
                        pchgNewId = newId
                        pchgOldId = oldId
                        pchgChange = change
                        pchgDist = newDists
            # now, do we have a potential change?
            if pchgCol is not None:
                trial[pchgCol] = pchgNewId
                score += pchgChange
                #adjust the dist matrix and contribution list
                #dist[pchgCol] = pchgDist[pchgCol+1:]#to items later than col
                for i in range(pchgCol+1,len(trial)):
                        pi = i - pchgCol - 1
                        contribution[i] += - dist[pchgCol][pi]
                        dist[pchgCol][pi] = pchgDist[i]
                        contribution[i] += pchgDist[i]
                for i in range(pchgCol):  #adjust distances to trial items before
                    contribution[i] += -dist[i][pchgCol-i-1]
                    dist[i][pchgCol-i-1] = pchgDist[i]
                    contribution[i] += pchgDist[i]
                contribution[pchgCol] = sum(pchgDist)
                if not math.isclose(score , sum(contribution)/2):
                    think()
                pchgCol = None # we've used up this change

            else:  # then this is local minimum
                if bestsofar is None or score < bestsofar:
                    besttrial = trial
                    bestsofar = score
                    bestcontribution = contribution
                break  # on to the next random trial

    # return the best trial and its score
    return besttrial, bestcontribution #bestsofar

maxids = maxids2

def angle_ids(i1, i2, cache):
    """
    here we compute the angle between two ids (which is of course the
    angle between their vectors), first checking the cache to see if we
    already did it once.
    """
    global meng, mara
    check = cache.get((i1,i2),None)
    if check is not None:
        return check
    #else:
    if i1[:2] != i2[:2]: complain('comparing arabic and english ids')
    if i1[:2] == 'ar':
        mod = mara
    elif i1[:2] == 'en':
        mod = meng
    else:
        complain('unknown language id:', i1)
    
    if rank_distance:
        if electra:
            retval = mod.rank_similarity_ide(i1, i2)
        else:
            retval = mod.rank_similarity_ids(i1, i2)
    else:
        if electra:
            v1 = mod.npvece[mod.ids[i1]]
            v2 = mod.npvece[mod.ids[i2]]
        else:
            v1 = mod.npvecs[mod.ids[i1]]
            v2 = mod.npvecs[mod.ids[i2]]
            
        retval = angle(v1/nl.norm(v1),v2/nl.norm(v2))

    # cache retval before returning
    cache[(i1,i2)] = cache[(i2,i1)] = retval
    return retval

def angle(v1, v2):
    """
    v1 and v2 are normalized vectors
    return the angle between them
    """
    cos = v1 @ v2
    # because the vectors are float32, occasionally there is a 
    #  conversion error caused by an unimportant difference
    #  in the low-order bits of the mantissa.
    # Insignificant in float32, but in the middle of the
    # mantissa for float64.
    # this can lead to a math domain error from math.acos()
    if cos > 1 and cos <1.0001: return 0

    return math.acos(cos)

main()
