from random import random
from typing import Tuple
from python_speech_features import mfcc
from os import listdir, path
from sys import argv
import scipy.io.wavfile as wav
import numpy as np
import pprint
import pickle
import operator
import json


def extract_feat(directory: str) -> Tuple[dict, list]:
	"""Generate class labels and feature vectors of audio files.

	Parameters
	-------------
	directory: relative path of directory containing GZTAN dataset

	Returns
	-------------
	labels: dict containing genre class labels
	dataset: list of feature vectors of all audio files
	"""

	with open("./feat.dat", "wb") as f, open("./labels.json", "w") as g:

		labels = {}
		dataset = []

		i = 0

		for i, folder in enumerate(listdir(directory), 1):
			print(f"Reading {folder}")

			if i == 11:  # Make sure only 10 dirs exist for 10 classes
				break

			labels[str(i)] = folder

			for file in listdir(path.join(directory, folder)):

				(rate, sig) = wav.read(path.join(directory, folder, file))
				mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)

				covariance = np.cov(np.matrix.transpose(mfcc_feat))
				mean_matrix = mfcc_feat.mean(0)
				feature = (mean_matrix, covariance, i)

				dataset.append(feature)

				pickle.dump(feature, f)

			print(f"Added feature matrices of {folder}")

		g.write(json.dumps(labels, indent=4))

	return labels, dataset


def split_data(filename: str, split: float) -> Tuple[list, list]:
	"""Loads the feature data from DAT file and splits it into train and test data. Single DAT file contains all the audio files.

	Parameters
	-------------
	filename: relative path of teh DAT file containing features
	split: train/test split ratio in decimal

	Returns
	-------------
	training_data: list
	test_data: list
	"""

	dataset = []
	training_data = []
	test_data = []

	with open(filename, "rb") as f:
		# Read dataset
		while True:
			try:
				dataset.append(pickle.load(f))
			except EOFError:
				f.close()
				break

	# Split training and test data
	for x in range(len(dataset)):
		if random() < split:
			training_data.append(dataset[x])
		else:
			test_data.append(dataset[x])

	return training_data, test_data


def get_score(test_data_len: int, train_data: list, test_data: list, k: int) -> float:
	"""Return model prediction score

	Parameters
	-------------
	test_data_len: length of test data array
	train_data: Split data for training
	test_data: Split data for testing
	k: integer value of k in KNN

	Returns
	-------------
	accuracy: float value of model accuracy
	"""

	predictions = []

	print("Calculating score")

	for x in range(test_data_len):

		predictions.append(
			get_nearest_class(get_neighbors(train_data, test_data[x], k))
		)

	accuracy = get_accuracy(test_data, predictions)

	return accuracy


def get_distance(instance_one, instance_two, k: int):
	"""Return distance between two instances

	Parameters
	-------------
	instance_one: feature of first audio file
	instance_two: feature of second audio file
	k: integer value of k in KNN

	Returns
	-------------
	distance: distance between two audio files
	"""

	mm1 = instance_one[0]
	cm1 = instance_one[1]

	mm2 = instance_two[0]
	cm2 = instance_two[1]

	distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
	distance += np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1)
	distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
	distance -= k

	# print(f"Distance is {distance}")

	return distance


def get_neighbors(train_data: list, instance, k: int) -> list:
	"""Return k nearest neighbors from an instance

	Parameters
	-------------
	train_data: train data containing feature vectors for training
	instance: feature vector for a single audio file
	k: integer value of k in KNN

	Returns
	-------------
	neighbors: list of k nearest neighbors to the instance
	"""

	distances = []
	neighbors = []

	for x in range(len(train_data)):
		dist = get_distance(train_data[x], instance, k) + get_distance(
			instance, train_data[x], k
		)
		distances.append((train_data[x][2], dist))

	distances.sort(key=operator.itemgetter(1))

	for x in range(k):
		neighbors.append(distances[x][0])

	return neighbors


def get_nearest_class(neighbors: list) -> int:
	"""Return nearest class based on vote

	Parameters
	-------------
	neighbors: list of k nearest neighbors

	Returns
	-------------
	label: int value of label
	"""

	class_vote = {}

	for x in range(len(neighbors)):

		response = neighbors[x]

		if response in class_vote:
			class_vote[response] += 1
		else:
			class_vote[response] = 1

	sorter = sorted(class_vote.items(), key=operator.itemgetter(1), reverse=True)

	return sorter[0][0]


def get_accuracy(test_data: list, predictions: list) -> float:
	"""Return accuracy of predictions

	Parameters
	-------------
	test_data: list of feature vectors for testing
	predictions: list of predictions on test data by model

	Returns
	-------------
	accuracy: float value of accuracy
	"""

	correct = 0

	for x in range(len(test_data)):
		if test_data[x][-1] == predictions[x]:
			correct += 1

	return 1.0 * correct / len(test_data)


def test_track(track_file: str) -> str:
	"""Function to predict class for random user-specified track using KNN

	Parameters
	-------------
	track_file: relative path to test audio file
	training_data: 100% of audio data

	Returns
	-------------
	class: predicted class of the track_file
	"""

	pp = pprint.PrettyPrinter(indent=4)

	labels = {}
	directory = "./genres_original"
	training_data = []

	print("Initializing")
	print("Preparing data")

	if not path.exists("./labels.json") and not path.exists("./feat.dat"):
		labels, training_data = extract_feat(directory)

	else:
		with open("./labels.json") as x:
			labels = json.load(x)

		with open("./feat.dat", "rb") as f:
			# Read dataset
			while True:
				try:
					training_data.append(pickle.load(f))
				except EOFError:
					f.close()
					break

	print(f"Reading test file: {track_file}")
	print(f"Labels:")
	pp.pprint(labels)

	# Generate track feat matrix
	(rate, sig) = wav.read(track_file)

	mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
	covariance = np.cov(np.matrix.transpose(mfcc_feat))
	mean_matrix = mfcc_feat.mean(0)
	track_instance = (mean_matrix, covariance)

	print("Calculating nearest class")

	temp = get_nearest_class(get_neighbors(training_data, track_instance, 10))

	print(f"File {track_file} is of genre {labels[str(temp)]}")


def main() -> None:
	"""Driver code"""

	if len(argv) == 1:
		if not path.exists("./labels.json") and not path.exists("./feat.dat"):
			labels = extract_feat("./genres_original")
		else:
			with open("./labels.json") as f:
				labels = json.load(f)

		training_data, test_data = split_data("./feat.dat", 0.66)

		model_prediction = get_score(len(test_data), training_data, test_data, 10)

		print(f"Model score: {model_prediction}")

	else:
		if argv[1] == "-t":
			test_track(argv[2])
		else:
			print("Incorrect flag")

	return None


if __name__ == "__main__":
	main()
