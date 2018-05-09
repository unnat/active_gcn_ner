from gcn_ner import GCNNer

if __name__ == '__main__':
    ner = GCNNer(ner_filename='./data/unlabeled_50/ner-gcn-9.tf', trans_prob_file='./data/trans_prob.pickle')
    ner.test('./data/labeled.conll')
