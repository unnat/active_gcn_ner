from gcn_ner import GCNNer
import sys

if __name__ == '__main__':
    # ner = GCNNer(ner_filename='./data/unlabeled_50/ner-gcn-9.tf', trans_prob_file='./data/trans_prob.pickle')
    ner = GCNNer(ner_filename='./data/{}/ner-gcn-{}.tf'.format(sys.argv[1], sys.argv[2]), trans_prob_file='./data/trans_prob.pickle')
    print('./data/{}/ner-gcn-{}.tf'.format(sys.argv[1], sys.argv[2]))
    ner.test('./data/dev.conll')