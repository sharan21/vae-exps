mkdir data
mkdir ./data/ptb

wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

tar -xf  simple-examples.tgz

mv simple-examples/data/ptb.train.txt ./data/ptb
mv simple-examples/data/ptb.valid.txt ./data/ptb
mv simple-examples/data/ptb.test.txt ./data/ptb

rm -rf simple_examples
