mkdir data
cd data
mkdir model_training
cd model_training
#--------------------------------------
# taskmaster
git clone https://github.com/google-research-datasets/Taskmaster.git
#--------------------------------------
# daily dialog
wget http://yanran.li/files/ijcnlp_dailydialog.zip
unzip ijcnlp_dailydialog.zip && rm ijcnlp_dailydialog.zip
#--------------------------------------
# empathetic dialogues
#wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
#tar -xvzf empatheticdialogues.tar.gz
#rm empatheticdialogues.tar.gz
#--------------------------------------
# personalized dialog
#wget https://www.dropbox.com/s/4i9u4y24pt3paba/personalized-dialog-dataset.tar.gz
#tar -xvzf personalized-dialog-dataset.tar.gz
#rm personalized-dialog-dataset.tar.gz
#--------------------------------------
# persona chat
wget https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json
