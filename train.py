"""
train.py

Faster R-CNN for gene prediction in the genome sequence.

@author: Venkata Pillutla
"""

from keras.utils import generic_utils
from keras.layers import Input
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import one_hot_encoding_sequences
from utils import losses
from models.baseNN import neuralnets
from models.baseNN import Config

from data_generation import data_generators
from data_generation.GenemeDataFetcher import read_genome_sequence_fromFile, getData, writeFeatureToDisk
import data_generation.DataGenerator
from data_generation import roi_helpers


import sys
import time


sys.setrecursionlimit(40000)

c = Config.Config()

#number of epochs
num_epochs = 4

#0 represents non gene, 1 represents gene 
classes_count = [0,1]


#splitting the sequence into n characters
seqlen = 10000

#Reading input data from file
print 'INFO: Reading input data from file'
wholeSequence = read_genome_sequence_fromFile('511145.12')

#print writeFeatureToDisk('511145.12')
dataFrame = getData('511145.12')


counter = 1
gene_location_dict = {}
for index, row in dataFrame.iterrows():
	if int(row['start']) <= (counter * seqlen) and int(row['end'])<= (counter * seqlen):
		if counter-1 not in gene_location_dict.keys():
			gene_location_dict[counter-1] = []
		gene_location_dict[counter-1].append((row['start'],row['end']))

	elif int(row['start']) <= (counter * seqlen) and (int(row['end']) > (counter * seqlen)):
		if counter-1 not in gene_location_dict.keys():
			gene_location_dict[counter-1] = []
		


		gene_location_dict[counter-1].append((row['start'],counter*seqlen))
		counter = counter + 1
		gene_location_dict[counter-1] = []
		gene_location_dict[counter-1].append(((counter-1)*seqlen, row['end']))
	elif int(row['start']) > (counter *seqlen):
		counter = counter +1
		if (int(row['end']) > (counter * seqlen)) and (int(row['end'] < (2 * counter * seqlen))):
			if counter-1 not in gene_location_dict.keys():
				gene_location_dict[counter-1] = []
			gene_location_dict[counter-1].append((row['start'], row['end']))

print 'INFO: gene_location_dict populated'

#whole sequence splitted into multiple smaller sequences.
sequence = []

for i in range(0, len(wholeSequence), seqlen):
	sequence.append(wholeSequence[i: i+seqlen])

class_count = {}
class_mapping = {}

allSequences = {}
try:
	#last element null - issue should be fixed instead of hard coding
	for seqid, seq in enumerate(sequence[0:-1]):
		#print 'currently processing '+str(seqid)+' of '+str(len(sequence)-1)+' sequences.'
		if seqid not in allSequences.keys():



			allSequences[seqid] = {}

			allSequences[seqid]['seqid'] = seqid
			a = []
			a.append(seq)
			a = one_hot_encoding_sequences(a, seqlen)
			a = np.array(a)
			
			a = a.reshape((1, seqlen, 4))
			
			allSequences[seqid]['sequence'] = a
			allSequences[seqid]['width'] = seqlen
			allSequences[seqid]['height'] = 1
			allSequences[seqid]['bboxes'] = []
			if np.random.randint(0,6) > 0:
				allSequences[seqid]['seqset'] = 'trainval'
			else:
				allSequences[seqid]['seqset'] = 'test'

			#adding ground truth bounding boxes
			if seqid in gene_location_dict.keys():
				if len(gene_location_dict[seqid]) == 0:
					print 'INFO: sequenceid '+ str(seqid) + ' doesnt contain a gene.'
					class_name = 'bg'
					if class_name not in class_count:
						class_count[class_name] = 1
					else:
						class_count[class_name] += 1

					if class_name not in class_mapping:
						class_mapping[class_name] = len(class_mapping)						
				else:
					for bbox in gene_location_dict[seqid]:
						class_name = 1
						if seqid > 1:
							allSequences[seqid]['bboxes'].append({'class': int(1), 'x1': int(bbox[0])-(seqid * seqlen) , 'x2': int(bbox[1])-(seqid * seqlen), 'y1': int(1),'y2': int(1)})
						else:
							allSequences[seqid]['bboxes'].append({'class': int(1), 'x1': int(bbox[0]) , 'x2': int(bbox[1]), 'y1': int(1),'y2': int(1)})

					if class_name not in class_count:
						class_count[class_name] = 1
					else:
						class_count[class_name] += 1

					if class_name not in class_mapping:
						class_mapping[class_name] = len(class_mapping)

			#allSequences[seqid]['bboxes'].append({'class': int(class_name), 'x1': int(x1) , 'x2': int(x2), 'y1': int(y1),'y2': int(y2)})

except Exception as e:
	print e

##each data point is a dictionary with seqid starting from 0, sequence, height = 1, width = seqlen, bboxes = list of single item dict

all_data = []
for key in allSequences:
	all_data.append(allSequences[key])

#all_data = np.array(all_data)


print 'INFO: data_generation step complete'

all_data_npy_format = np.array(all_data)
print 'INFO: Storing data on to disk'



np.save('sample_data/data_set.npy', all_data_npy_format)



#all_data = np.load('sample_data/data_set.npy')

#print all_data.shape
#print 'sequence', all_data[0]['sequence']
#print type(all_data)

data_gen_train = data_generators.get_anchor_gt(all_data, class_count, c, neuralnets.get_img_output_length, K.image_dim_ordering(), mode='train')


#input
#sequence = ['atgctgatagctgattatgcgatctagtga','atgctgatagctgctacgtcgttcatgatg']

sequenceWidth = len(sequence[0])


#converting the sequences to one-hot vector representation
print 'INFO: Converting splitted sequences into one-hot representation'
X = np.array(one_hot_encoding_sequences(sequence, seqlen))

print 'INFO: input_shape ',X.shape


X = X.reshape(X.shape[0], 1, sequenceWidth, 4)
print X.shape

#y - yet to be created


#only the first element
#var = K.variable(value = X[0].reshape((1,1,sequenceWidth,4)))

#1, sequenceWidth, 4 (4 channels after one-hot encoding atgc)
genome_shape = (1, sequenceWidth, 4)

# genome sequence input has 4 channels since the images are one hot encoded with 'atgc'
genome_sequence_input = Input(shape = genome_shape)

"""
#handling the error: expect  0 input but recieved 1 (https://github.com/fchollet/keras/issues/6155)
if genome_sequence_input is None:
	print 'DEBUG: input tensor is none.'
	img_input = Input(shape=input_shape)
else:
	if not K.is_keras_tensor(genome_sequence_input):
		print "DEBUG: Input not a keras tensor. Converting it to Keras Tensor"
		img_input = Input(tensor = genome_sequence_input, shape=genome_shape)
	else:
		img_input = genome_sequence_input
"""




roi_input = Input(shape = (c.num_rois, 4))



number_of_anchors = len(c.anchor_box_scales) * len(c.anchor_box_ratios)

#define the neural network here
shared_layers = neuralnets.nn_base(genome_sequence_input)


#define the Region proposal network here - TODO
rpn = neuralnets.regionProposalNetwork(shared_layers, number_of_anchors)



classifier = neuralnets.classifier(shared_layers, roi_input, c.num_rois, nb_classes=len(classes_count), trainable=True)


model_rpn = Model(genome_sequence_input, rpn[:2])
#model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([genome_sequence_input, roi_input], classifier)


# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([genome_sequence_input, roi_input], rpn[:2] + classifier)


optimizer = Adam(lr=1e-4)
optimizer_classifier = Adam(lr=1e-4)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(number_of_anchors), losses.rpn_loss_regr(number_of_anchors)])


model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 1000
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf


print model_rpn.inputs

print "-----Training Start-----"

print model_rpn.summary()

for epoch_num in range(num_epochs):

	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
	
	#while True:
	#for i in range(0,100):
	for X, Y, img_data in data_gen_train:
		
		if len(rpn_accuracy_rpn_monitor) == epoch_length and c.verbose:
			mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
			rpn_accuracy_rpn_monitor = []
			print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
			if mean_overlapping_bboxes == 0:
				print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

		#X, Y, img_data = next(data_gen_train)

		X = X.reshape(1, 1, 10000, 4)
		#print 'DEBUG: Before error: x shape, y shape', X.shape, len(Y)
		#print Y[0].shape, Y[1].shape


		loss_rpn = model_rpn.train_on_batch(X, Y)
		#loss_rpn = model_rpn.fit(X, Y)

		P_rpn = model_rpn.predict_on_batch(X)

		
		#print P_rpn[0].shape
		#print P_rpn[1].shape

		R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], c, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

		# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
		X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, c, class_mapping)

		if X2 is None:
			rpn_accuracy_rpn_monitor.append(0)
			rpn_accuracy_for_epoch.append(0)
			continue

		neg_samples = np.where(Y1[0, :, -1] == 1)
		pos_samples = np.where(Y1[0, :, -1] == 0)

		if len(neg_samples) > 0:
			neg_samples = neg_samples[0]
		else:
			neg_samples = []

		if len(pos_samples) > 0:
			pos_samples = pos_samples[0]
		else:
			pos_samples = []

		rpn_accuracy_rpn_monitor.append(len(pos_samples))
		rpn_accuracy_for_epoch.append((len(pos_samples)))

		if c.num_rois > 1:
			if len(pos_samples) < c.num_rois//2:
				selected_pos_samples = pos_samples.tolist()
			else:
				selected_pos_samples = np.random.choice(pos_samples, c.num_rois//2, replace=False).tolist()
			try:
				selected_neg_samples = np.random.choice(neg_samples, c.num_rois - len(selected_pos_samples), replace=False).tolist()
			except:
				selected_neg_samples = np.random.choice(neg_samples, c.num_rois - len(selected_pos_samples), replace=True).tolist()

			sel_samples = selected_pos_samples + selected_neg_samples
		else:
			# in the extreme case where num_rois = 1, we pick a random pos or neg sample
			selected_pos_samples = pos_samples.tolist()
			selected_neg_samples = neg_samples.tolist()
			if np.random.randint(0, 2):
				sel_samples = random.choice(neg_samples)
			else:
				sel_samples = random.choice(pos_samples)

		loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

		losses[iter_num, 0] = loss_rpn[1]
		losses[iter_num, 1] = loss_rpn[2]

		losses[iter_num, 2] = loss_class[1]
		losses[iter_num, 3] = loss_class[2]
		losses[iter_num, 4] = loss_class[3]

		iter_num += 1

		progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
								  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

		if iter_num == epoch_length:
			loss_rpn_cls = np.mean(losses[:, 0])
			loss_rpn_regr = np.mean(losses[:, 1])
			loss_class_cls = np.mean(losses[:, 2])
			loss_class_regr = np.mean(losses[:, 3])
			class_acc = np.mean(losses[:, 4])

			mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
			rpn_accuracy_for_epoch = []

			if C.verbose:
				print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
				print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
				print('Loss RPN classifier: {}'.format(loss_rpn_cls))
				print('Loss RPN regression: {}'.format(loss_rpn_regr))
				print('Loss Detector classifier: {}'.format(loss_class_cls))
				print('Loss Detector regression: {}'.format(loss_class_regr))
				print('Elapsed time: {}'.format(time.time() - start_time))

			curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
			iter_num = 0
			start_time = time.time()

			if curr_loss < best_loss:
				if C.verbose:
					print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
				best_loss = curr_loss
				model_all.save_weights(c.model_path)

			break





