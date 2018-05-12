CS410 project: Active Learning for GCN based Named Entity Recognition
=============================

This is our modified implementation of Named Entity Recognizer that can
actively learn on limited initial annotated data. The original Graph
Convolutional Network model for named entity recognition can be found
in [this repository](https://github.com/ContextScout/gcn_ner) and the 
reference article for the base model is [Graph Convolutional
Networks for Named Entity
Recognition](https://arxiv.org/abs/1709.10053).

In this work, we investigated a yet unexplored aspect of actively learning for text information systems. We chose the task of parsing dependencies for Named Entity Recognition (NER). We evaluate the impact of Graph Convolutional Networks (GCNs) for active learning based NER. We find the GCNs are indeed effective in indicating which unlabeled examples should we uncover labels for. Uncertainty sampling proves more effective than random sampling in improving performance over the unseen test set. 

Due to computational and time resources, we limited the scope of this project. Going forward, we would want to use other text information tasks (beyond NER) as the test bed of for our approach. Also we would be excited to explore more sophisticated active learning methodologies.

The system currently uses the word vectors that come with spacy's "en_core_web_md" model.

Installation
------------
```bash
git clone https://github.com/contextscout/gcn_ner.git

cd gcn_ner

virtualenv --python=/usr/bin/python3 .env

source .env/bin/activate

pip install -r requirements.txt

python -m spacy download en

python -m spacy download en_core_web_md
```

if you want to install Tensorflow with GPU capabilities please use 
```python
pip install -r requirements_gpu.txt
```

Data
--------
As the datasets are copyrighted, we can provide the access to course staff on request.


Test NER on a text
--------
Execute the file
```python
python test_ner.py < data/random_text.txt
```


Train NER from a dataset
---------
You will need to put your 'train.conll' into the 'data/' directory,
then execute the file
```python
python train.py
```


Test the dataset F1 score
-------------------------
You will need to put your 'dev.conll' or 'test.conll' into the 'data/' directory,
then execute the file
```python
python test_dataset.py
```

CONLL format
------------

The training/testing conll files are in the conll format, as in the following example. 
Only the fourth, fifth, and eleventh columns are used.

```code
source_file_name   1    0              New   NNP    (TOP(S(NP*         -    -   -   Speaker#1    (GPE*      *       (ARG1*   (ARG1*   (19
source_file_name   1    1             York   NNP             *)        -    -   -   Speaker#1        *)     *            *)       *)   19)
source_file_name   1    2              was   VBD          (VP*         be  03   -   Speaker#1        *    (V*)           *        *     -
source_file_name   1    3        developed   VBN          (VP*    develop  02   -   Speaker#1        *      *          (V*)       *     -
source_file_name   1    4             from    IN          (PP*         -    -   -   Speaker#1        *      *       (ARG2*        *     -
source_file_name   1    5                a    DT          (NP*         -    -   -   Speaker#1        *      *            *        *     -
source_file_name   1    6          hunting    NN             *         -    -   -   Speaker#1        *      *            *        *     -
source_file_name   1    7           harbor    NN            *))        -    -   -   Speaker#1        *      *            *)       *     -
source_file_name   1    8              one    CD  (ADVP(NP(QP*         -    -   -   Speaker#1   (DATE*      *   (ARGM-TMP*        *     -
source_file_name   1    9          million    CD             *)        -    -   -   Speaker#1        *      *            *        *     -
source_file_name   1   10            years   NNS             *)        -    -   -   Speaker#1        *      *            *        *     -
source_file_name   1   11              ago    RB             *)        -    -   -   Speaker#1        *)     *            *)       *     -
source_file_name   1   12               to    TO        (S(VP*         -    -   -   Speaker#1        *      *   (ARGM-PRP*        *     -
source_file_name   1   13           become    VB          (VP*     become  01   1   Speaker#1        *      *            *      (V*)    -
source_file_name   1   14            today    NN       (NP(NP*         -    -   -   Speaker#1    (DATE)     *            *   (ARG2*     -
source_file_name   1   15               's   POS             *)        -    -   -   Speaker#1        *      *            *        *     -
source_file_name   1   16    international    JJ             *         -    -   -   Speaker#1        *      *            *        *     -
source_file_name   1   17       metropolis   NNS        *))))))        -    -   -   Speaker#1        *      *            *)       *)    -
```


