from gcn_ner import GCNNer

if __name__ == '__main__':
	
	from numpy import genfromtxt
	import numpy as np
	
	# al_length = 3750
	# al_list = list(np.random.randint(0,45112,al_length))
	# GCNNer.train_and_save(dataset='./data/labeled.conll', saving_dir='./data/unlabeled_50_random', epochs=20, al_args=al_list, load_ckpt="./data/unlabeled_50/ner-gcn-9.tf")

	# my_data = genfromtxt('unlabeled_50_scores_sorted.csv', delimiter=',')
	# al_length = 3750
	# al_list = list(my_data[:3750,0].astype(np.int))
	# print("Total finetuning samples: {}".format(len(al_list)))
	# GCNNer.train_and_save(dataset='./data/labeled.conll', saving_dir='./data/unlabeled_50_uncertain_2', epochs=20, al_args=al_list, load_ckpt="./data/unlabeled_50/ner-gcn-9.tf")


	my_data = genfromtxt('unlabeled_50_scores_sorted.csv', delimiter=',')
	al_length = 3750
	al_list = list(my_data[:3750,0].astype(np.int))
	al_list.extend(range(45112, 45112+15177))
	print("Total finetuning samples: {}".format(len(al_list)))
	GCNNer.train_and_save(dataset='./data/labeled_and_unlabeled_50.conll', saving_dir='./data/unlabeled_50_uncertain_combined', epochs=20, al_args=al_list, load_ckpt="./data/unlabeled_50/ner-gcn-9.tf")

	# al_length = 3750
	# al_list = list(np.random.randint(0,45112,al_length))
	# al_list.extend(range(45112, 45112+15177))
	# print("Total finetuning samples: {}".format(len(al_list)))
	# GCNNer.train_and_save(dataset='./data/labeled_and_unlabeled_50.conll', saving_dir='./data/unlabeled_50_random_combined', epochs=20, al_args=al_list, load_ckpt="./data/unlabeled_50/ner-gcn-9.tf")

	# my_data = genfromtxt('unlabeled_50_scores_sorted.csv', delimiter=',')
	# al_length = 3750
	# al_list = list(my_data[:3750,0].astype(np.int))
	# al_list.extend(range(45112, 45112+15177))
	# print("Total finetuning samples: [UC] {}".format(len(al_list)))
	# GCNNer.train_and_save(dataset='./data/labeled_and_unlabeled_50.conll', saving_dir='./data/unlabeled_50_uncertain_combined_scratch', epochs=30, al_args=al_list)

	# al_length = 3750
	# al_list = list(np.random.randint(0,45112,al_length))
	# al_list.extend(range(45112, 45112+15177))
	# print("Total finetuning samples: [Random] {}".format(len(al_list)))
	# GCNNer.train_and_save(dataset='./data/labeled_and_unlabeled_50.conll', saving_dir='./data/unlabeled_50_random_combined_scratch', epochs=30, al_args=al_list)