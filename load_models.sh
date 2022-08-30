# bart-base files
mkdir -p facebook/bart-base-local
cd facebook/bart-base-local
wget https://cdn.huggingface.co/facebook/bart-base/pytorch_model.bin
wget https://cdn.huggingface.co/facebook/bart-base/config.json
wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt
wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json

# gpt2 files
#mkdir gpt2-local
#cd gpt2-local
#wget -O pytorch_model.bin https://cdn.huggingface.co/gpt2-pytorch_model.bin
#wget -O config.json https://cdn.huggingface.co/gpt2-config.json
#wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
#wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json