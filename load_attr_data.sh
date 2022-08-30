mkdir data
cd data
mkdir discriminator_finetuning
cd discriminator_finetuning
#--------------------------------------
# SST-5
mkdir sentiment
cd sentiment
wget https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/sst_train.txt
wget https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/sst_test.txt
wget https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/sst_dev.txt
#--------------------------------------