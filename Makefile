
all:	sub1e.zip sub1s.zip sub2e.zip sub2s.zip

sub1e.zip:	output1e.json
	zip sub1e.zip  output1e.json	

sub2e.zip:	output2e.json
	zip sub2e.zip  output2e.json	

output1e.json:	trainRD.py readCLRD.py
	python3 trainRD.py 1 E

output2e.json:	trainRD.py readCLRD.py
	python3 trainRD.py 2 E


sub1s.zip:	output1s.json
	zip sub1s.zip  output1s.json	

sub2s.zip:	output2s.json
	zip sub2s.zip  output2s.json	

output1s.json:	trainRD.py readCLRD.py
	python3 trainRD.py 1 S

output2s.json:	trainRD.py readCLRD.py
	python3 trainRD.py 2 S

graphs:	
	gnuplot results.gnuplot

#results/ara-sgns-scatter-cos.vs.rank.png:
#results/ara-Electra-scatter-cos.vs.rank.png:
#results/eng-sgns-scatter-cos.vs.rank.png:
#results/eng-Electra-scatter-cos.vs.rank.png:
