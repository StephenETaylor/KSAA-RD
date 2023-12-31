This directory contains the code for the UWB submission to the KSAA-RD task
for ArabicNLP-2023

The KSAA Reverse Dictionary task asks participants to produce a word-embedding which should correspond to a sense of an Arabic word, based on [task 1] an Arabic gloss or [task2] an English gloss.  

The training data provides:
> skipgram word embeddings ('sgns') which are the same for all senses of a word.

 and 

>electra context embeddings, which depend upon the context of the word, and might not correspond exactly to the sense, since each sense can be used in many different contexts.

>> The data for the task is or will be posted on https://github.com/Waadtss/ArReverseDictionary 
>> The baseline implementation and the scoring program for the task can also be
   found there.

This system beats the baseline only on Arabic definitions and skipgram embeddings.

File descriptions

>Makefile prepares submissions for CodaLab:

1) for subtask 1

  https://codalab.lisn.upsaclay.fr/competitions/14568#participate-submit_results

>>zip files sub1s.zip and sub1e.zip are for task1, sgns and electra, respectively, and are prepared from output1s.json or output1d.json with the zip application, because codalab wants its submissions in zip format.


2) for subtask 2

  https://codalab.lisn.upsaclay.fr/competitions/14569#participate-submit_results

>>zip files sub2s.zip and sub2e.zip are for task2, sgns and electra, respectively, and are prepared from output2s.json or output2d.json with the zip application, because codalab wants its submissions in zip format.


>readCLRD.py reads through the various input files and digests the data into
python pickle files.

>trainRD.py reads through the pickle files prepared by readCLRD.  
It defaults to task 1, Electra embedding, but depending on its command line,
 can also perform the other variations on the task, to wit:
 >> python3 trainrd.py 1 e   \

 >>> read ar.test.json, each record of which contains an "id" and a "gloss" key.
 >>> and produce output1e.json, each record of which  will contain an "id" and an "electra" key

     
 >> python3 trainrd.py 1 s   \

 >>> read ar.test.json, each record of which contains an "id" and a "gloss" key.
 >>> and produce output1s.json, each record of which  will contain an "id" and an "sgns" key

 >> python3 trainrd.py 2 e   \

 >>> read ar.en.test.json, each record of which contains an "id" and a "gloss" key.
 >>> and produce output2e.json, each record of which  will contain an "id" and an "electra" key

     
 >> python3 trainrd.py 2 s   \

 >>> read ar.en.test.json, each record of which contains an "id" and a "gloss" key.
 >>> and produce output2s.json, each record of which  will contain an "id" and an "sgns" key


     
    

    

     
    

    

