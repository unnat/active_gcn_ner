from gcn_ner import GCNNer

if __name__ == '__main__':
	
	# from numpy import genfromtxt
	# import numpy as np
	# my_data = genfromtxt('unlabeled_50_scores_sorted.csv', delimiter=',')
	# al_length = 3750
	# al_list = list(my_data[:3750,0].astype(np.int))

	# GCNNer.train_and_save(dataset='./data/labeled.conll', saving_dir='./data/unlabeled_50_uncertain', epochs=10, al_args=al_list, load_ckpt="./data/unlabeled_50/ner-gcn-9.tf")

	al_fraction = 0.08312644085
	GCNNer.train_and_save(dataset='./data/labeled.conll', saving_dir='./data/unlabeled_50_random', epochs=10, al_args=al_fraction, load_ckpt="./data/unlabeled_50/ner-gcn-9.tf")
