import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def game_state_to_data_sample(game_state: dict, bounds, block_size):
	snake_head = game_state["snake_body"][-1]  # coordinates of the snake's head
	snake_tail = game_state["snake_body"][0:-1]  # the rest of the snake's body
	food = game_state["food"]  # current food coordinates
	last_move = np.zeros(4)

	# last move - attribute
	LastMove = game_state["snake_direction"].value
	last_move[LastMove] = 1

	# determine the coordinates of neighbour cells
	up_coordinates = (snake_head[0], snake_head[1] - block_size)
	down_coordinates = (snake_head[0], snake_head[1] + block_size)
	left_coordinates = (snake_head[0] - block_size, snake_head[1])
	right_coordinates = (snake_head[0] + block_size, snake_head[1])

	# check for food in neighbour cells - attributes
	FoodUp = 1 if up_coordinates == food else 0
	FoodDown = 1 if down_coordinates == food else 0
	FoodLeft = 1 if left_coordinates == food else 0
	FoodRight = 1 if right_coordinates == food else 0

	# check for food in each direction - attributes
	FoodDirUp = 1 if food[0] == snake_head[0] and food[1] < snake_head[1] else 0
	FoodDirDown = 1 if food[0] == snake_head[0] and food[1] > snake_head[1] else 0
	FoodDirLeft = 1 if food[0] < snake_head[0] and food[1] == snake_head[1] else 0
	FoodDirRight = 1 if food[0] > snake_head[0] and food[1] == snake_head[1] else 0

	# check for obstacles in neighbour cells - attributes
	ObstacleUp = 1 if up_coordinates in snake_tail or snake_head[1] == 0 else 0
	ObstacleDown = 1 if down_coordinates in snake_tail or snake_head[1] + block_size == bounds[1] else 0
	ObstacleLeft = 1 if left_coordinates in snake_tail or snake_head[0] == 0 else 0
	ObstacleRight = 1 if right_coordinates in snake_tail or snake_head[0] + block_size == bounds[0] else 0

	# return an array of attributes
	return np.hstack((np.array(
		[FoodUp, FoodDirUp, FoodDown, FoodDirDown, FoodLeft, FoodDirLeft, FoodRight, FoodDirRight, ObstacleUp,
		 ObstacleDown, ObstacleLeft, ObstacleRight], dtype=int), last_move)).reshape(1, 16)


def create_dataset(file):
	# initialize data arrays
	dataset = np.zeros((1, 16), dtype=int)
	decisions = np.zeros((1, 4), dtype=int)

	# load data from file
	with open(file, 'rb') as f:
		data_file = pickle.load(f)

	# interpret data
	states = data_file["data"]
	bounds = data_file["bounds"]
	block_size = data_file["block_size"]
	# assign attributes to each state
	for state in states:
		decision = np.zeros((1, 4), dtype=int)
		dataset = np.append(dataset, game_state_to_data_sample(state[0], bounds, block_size), axis=0)
		decision[0][[[state[1].value]]] = 1
		decisions = np.append(decisions, decision, axis=0)
	return dataset[1:], decisions[1:]


def prepare_dataset():
	train_split = 0.8

	# import datasets with attributes
	# dataset, decisions = create_dataset(f"./2024-05-19_18-54-56.pickle")
	dataset_1, decisions_1 = create_dataset(f"./2024-05-18_19-38-16.pickle")
	dataset_2, decisions_2 = create_dataset(f"./2024-05-19_18-54-56.pickle")

	dataset = np.vstack((dataset_1, dataset_2))
	decisions = np.vstack((decisions_1, decisions_2))

	train_data, temp_data, train_decisions, temp_decisions = train_test_split(
		dataset, decisions, train_size=train_split, random_state=42)

	# Split the temporary set into validation and test sets equally
	val_data, test_data, val_decisions, test_decisions = train_test_split(
		temp_data, temp_decisions, test_size=0.5, random_state=42)

	# Create datasets
	train_dataset = BCDataset(train_data, train_decisions.tolist())
	val_dataset = BCDataset(val_data, val_decisions.tolist())
	test_dataset = BCDataset(test_data, test_decisions.tolist())

	return train_dataset, val_dataset, test_dataset

class BCDataset(Dataset):
	def __init__(self, dataset, decisions):
		self.input_data = torch.tensor(dataset, dtype=torch.float32)
		# print(decisions)
		# print(decisions.shape)
		self.decisions = torch.tensor(decisions, dtype=torch.float32).squeeze()
		# print(self.decisions)
		# print(self.decisions.shape)

	def __len__(self):
		return len(self.input_data)

	def __getitem__(self, index):
		item_attributes = self.input_data[index]
		item_decision = self.decisions[index]
		return item_attributes, item_decision
