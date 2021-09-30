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
python experiment.py -d <train_data_filename> -m palo -c ./config.ini -f <train_data_filename> train --epochs <num_epochs> --batch <batch_size>
```

### Predict
```bash
python ./predict.py -d <tmp_file> -m palo -c config -f <latest_model> 
```

### Batch Predict (on next version)
files=(...)
for i in ${files[@]}; do     echo "Predict on: $i"; python ./predict.py -d "$i" -m palo_ti -c config.ini -f model_registry/*.h5 ; done 

### Todos
[ ] Update dependencies
[ ] Check that batch predict works