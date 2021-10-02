# Polus
Polus: Sentiment Analysis for Greek Text based on *"Leveraging aspect-based sentiment prediction with textual features and document metadata"* paper for SETN2020. A Bi-LSTM neural network with 2 inputs (text and emojis map) for 3-class sentiment classification.

***python version: Python 3.6.9***


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

Train one of the following models:
* Polus TE model (train on text and emojis) ***-m polus_te***
* Polus TO model (train on text only) ***-m polus_to***

***config.ini: default params***
```bash
python polus.py -d <train_data_filename> -m <polus_model> -c ./config.ini -f <model_filename> train --epochs <num_epochs> --batch <batch_size>
```
example:
```text
python polus.py -d train_data.json -m polus_te -c ./config.ini -f te_model_ train --epochs <num_epochs> --batch <batch_size>
```

### Test
```bash
python polus.py -d <train_data_filename> -m palo_te -c ./config.ini -f test
```
example:
```text
python polus.py -d test_data.json -m palo_te -c ./config.ini -f test
```

### Todos

[ ] Update dependencies

[ ] Include predict
