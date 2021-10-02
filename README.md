# Polus (Under Development)
Polus: Sentiment Analysis for Greek Text based on *"Leveraging aspect-based sentiment prediction with textual features and document metadata"* paper for SETN2020.

#### Clone Repo
```bash
git clone https://github.com/kkorovesis/polus.git
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download embeddings
```bash
aws s3 cp s3://com.palo.ml-dev/embeddings/fasttext/cc.el.300.vec.gz embeddings/cc.el.300.vec.gz
gzip -d embeddings/cc.el.300.vec.gz
mv embeddings/cc.el.300.vec embeddings/fasttext.300d.txt
```

### Train
```bash
python polus.py -d <train_data_filename> -m palo_te -c ./config.ini -f <train_data_filename> train --epochs <num_epochs> --batch <batch_size>
```

### Predict
```bash
python ./predict.py -d <tmp_file> -m palo -c config -f <latest_model> 
```

s3://com.palo.ml-dev/polus/tree/banks/eurobank/data/TRAIN_ba4a4a87-f9e1-4e43-b7ef-0e8f3bad87af.json.gz

### Todos

[ ] Update dependencies

[ ] Check that batch predict works
