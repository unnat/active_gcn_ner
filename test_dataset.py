from gcn_ner import GCNNer
import sys

if __name__ == '__main__':
    # ner = GCNNer(ner_filename='./data/unlabeled_50/ner-gcn-9.tf', trans_prob_file='./data/trans_prob.pickle')
    ner = GCNNer(ner_filename='./data/unlabeled_50_uncertain/ner-gcn-{}.tf'.format(sys.argv[1]), trans_prob_file='./data/trans_prob.pickle')
    print('./data/unlabeled_50_uncertain/ner-gcn-{}.tf'.format(sys.argv[1]))
    ner.test('./data/dev.conll')
