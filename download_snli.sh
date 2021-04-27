mkdir data
mkdir ./data/snli

wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip -O data/snli/data.zip

unzip data/snli/data -C data/snli
mv data/snli/snli_1.0/* ./data/snli
rm -r data/snli/snli_1.0/
rm data/snli/data.zip