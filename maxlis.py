"""
separate work on maxids, which takes a long time to run.

"""
import random
import time

def maxids(lis, angleCache):
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
    #angleCache = dict()
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

import time
start = time.time()
def ctime(s):
    global start
    now = time.time()
    
    print(s,now-start, flush = True)
    start = now

def angle_ids(a,b,c):
    if (a,b) in c:
        return c[(a,b)]
    squawk()

def nfs(f,n):
    return ([f+str(i) for i in range(n)])

def maxids1(lis, angle_cache):
    for l1, s in genlis1(lis, None, angle_cache):
        pass
    return l1, s

    
def genlis1(lis,bestsofar, angle_cache):
    """
    A divide and conquer approach
    This is a generator, which returns a list of ids, one from each element of lis
    and the sum of the angles between all of them.  smaller angles are better
    parameters:
      lis is a list of lists of ids, e.g.
        [[ar2,ar4], [ar5, ar8], [ar1, ar3, ar6]]
      bestsofar is an angle, or none; we hope to short-circuit returning
        lists with angles >= bestsofar
      angle_cache is a dict containing angles between (idx, idy) pairs.

    The yielded value is a pair of a list and the sum of the angles between 
      every id in the list, e.g.
         [ar4, ar5, ar6], sum_of_3_angles

    We divide the list in half, generate on the two halves, then
    combine them, and finish up the missing angles.
    """
    ll = len(lis)
    if ll == 0: squawk()
    if ll == 1:
        for idv in lis[0]:
            yield (idv,0)
        return
    ll0 = ll//2
    ll1 = ll - ll0
    for half, first in genlis1(lis[:ll0], bestsofar, angle_cache):
        if bestsofar is not None and first > bestsofar: continue
        if bestsofar is not None:
            bsf2 = bestsofar - first
        else:
            bsf2 = None
        for rest, second in genlis1(lis[ll0:], bsf2,angle_cache):
            score1 = first + second
            for i1 in half:
                score2 = score1
                for i2 in rest:
                    score2 += angle_cache[(i1,i2)]
                    if bestsofar is not None and score2 > bestsofar:
                        score2 = score1
                        continue

            if bestsofar is None or score2 < bestsofar:
                bestsofar = score2
            yield half[:] + rest, score2

def maxids2(lis, cache):
    """
       A probalistic search
       for Most Likely Sense-ids
       assumption is that the most compatible senses are the correct ones for
       the gloss. 
       Here we randomly pick many times a set of senses, and then search
       nearby sets for a local minimum.
       return the best of these as an approximation to the best set of ids
    """
    bestsofar = None   # a final return value.  bestrial produced it.
    for _ in range(100):  # aim for a set composition in the 99TH Percentile
        trial = [] 
        # build random trial list of senses
        for idvs in lis:
            trial.append(idvs[random.randint(1,len(idvs))-1])
        # now score this set
        score = 0
        for i in range(len(lis)-1):
            for j in range(i+1,len(lis)):
                score += angle_ids(trial[i],trial[j], cache)
        # look for a possible change
        pchgCol = pchgNewId = pchgOldId = pchgChange = None
        while True:  # until we find a local minimum
            # examine each column of trial, and 
            for col in range(len(lis)):
                oldId = trial[col]

                #compute contribution of current occupant of column to score
                oldDiff = 0
                for i in range(len(trial)):
                    if i == col: continue
                    oldDiff += angle_ids(oldId,trial[i],cache)

                # now consider ids allowed for this column
                for newId in lis[col]:
                    #compute newId's contribution to the new score
                    newDiff = 0
                    for i in range(len(lis)):
                        if i == col: continue
                        newDiff += angle_ids(newId,trial[i],cache)

                    # does this change make score smaller?
                    if newDiff >= oldDiff: continue # would be worse
                    change = oldDiff - newDiff # so change is negative
                    if pchgChange is None or  change < pchgChange: 
                        # save the bestpotential change until all are examined 
                        pchgCol = col
                        pchgNewId = newId
                        pchgOldId = oldId
                        pchgChange = change
            # now, do we have a potential change?
            if pchgCol is not None:
                trial[pchgCol] = pchgNewId
                score += pchgChange
                pchgCol = None # we've used up this change

            else:  # then this is local minimum
                if bestsofar is None or score < bestsofar:
                    besttrial = trial
                    bestsofar = score
                break  # on to the next random trial

    # return the best trial and its score
    return besttrial, bestsofar

def angle_ids(l1, l2, cache):
    """
    This version of angle_ids returns random numbers for non-identical
    ids, and 0 for identical ones, but it caches the values, so that
    the same pair always returns the same value.
    """
    if l1 == l2: return 0
    if (l1,l2) in cache:
        return cache[l1,l2]

    # otherwise invent and save a comparison value
    retval = random.random()
    cache[(l1,l2)] = cache[(l2,l1)] = retval
    return retval




if __name__ == '__main__':

    test = 2

    cache = dict()
    cache[('f','f')] = 0
    start = time.time()
    f2 = nfs('f',2)
    for i in range(26):
        if test == 2:
            ll = [nfs(chr(ord('a')+x), 4) for x in range(i)]
        else:
            ll = [f2]*i
        
        if test == 0:
            maxids(ll,cache)
        elif test == 1:
            maxids1(ll, cache)
        elif test == 2:
            maxids2(ll, cache)
        ctime(str(i)+' '+str(2**i))


