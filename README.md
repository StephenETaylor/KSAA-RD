This directory contains the code for the UWB submission to the KSAA-RD task
for ArabicNLP-2023

>> The data for the task is or will be posted on https://github.com/Waadtss/ArReverseDictionary 
>> The baseline implementation and the scoring program for the task can also be
   found there.

The Makefile prepares submissions for CodaLab:

1) for subtask 1

  https://codalab.lisn.upsaclay.fr/competitions/14568#participate-submit_results

>>zip files sub1s.zip and sub1e.zip are for task1, sgns and electra, respectively

2) for subtask 2

  https://codalab.lisn.upsaclay.fr/competitions/14569#participate-submit_results

>>zip files sub2s.zip and sub2e.zip are for task2, sgns and electra, respectively


readCLRD.py reads through the various input files and digests the data into
python pickle files.

trainRD.py reads through the pickle files.  It defaults to 
 task 1, Electra embedding, but depending on its command line,
 can also perform the other variations on the task, to wit:
 >> python3 trainrd.py 1 e   \

 >>> read ar.test.json, each record of which contains an "id" and a "gloss" key.
 >>> and produce output1e.json, each record of which  will contain id "id" and an "electra" key

     
 >> python3 trainrd.py 1 s   \

 >>> read ar.test.json, each record of which contains an "id" and a "gloss" key.
 >>> and produce output1s.json, each record of which  will contain id "id" and an "SGNS" key

 >> python3 trainrd.py 2 e   \

 >>> read ar.test.json, each record of which contains an "id" and a "gloss" key.
 >>> and produce output2e.json, each record of which  will contain id "id" and an "electra" key

     
 >> python3 trainrd.py 2 s   \

 >>> read ar.test.json, each record of which contains an "id" and a "gloss" key.
 >>> and produce output2s.json, each record of which  will contain id "id" and an "SGNS" key

     
    

    

     
    

    

