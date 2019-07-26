make
time ./word2vec -train text8 -output vectors.txt -cbow 1 -size 50 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15
./distance vectors.txt