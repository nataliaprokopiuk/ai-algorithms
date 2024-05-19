import pickle
import numpy as np

def game_state_to_data_sample(game_state: dict, bounds, block_size):
	snake_head = game_state["snake_body"][-1] # coordinates of the snake's head
	snake_tail = game_state["snake_body"][0:-1] # the rest of the snake's body
	food = game_state["food"] # current food coordinates

    # last move - attribute
	LastMove = game_state["snake_direction"].value

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
	return np.array([FoodUp, FoodDirUp, FoodDown, FoodDirDown, FoodLeft, FoodDirLeft, FoodRight, FoodDirRight, ObstacleUp, ObstacleDown, ObstacleLeft, ObstacleRight, LastMove], dtype=int).reshape(1,13)

def create_dataset(file):
	# initialize data arrays
	dataset = np.zeros((1,13), dtype=int)
	decisions = np.zeros((1,1), dtype=int)

	# load data from file
	with open(file, 'rb') as f:
		data_file = pickle.load(f)
	
	# interpret data
	states = data_file["data"]
	bounds = data_file["bounds"]
	block_size = data_file["block_size"]
	# assign attributes to each state
	for state in states:
		# print(state)
		# print(game_state_to_data_sample(state[0], bounds, block_size))
		# print()
		dataset = np.append(dataset, game_state_to_data_sample(state[0], bounds, block_size), axis=0)
		decisions = np.append(decisions, [[state[1].value]], axis=0)

	return dataset[1:], decisions[1:]

