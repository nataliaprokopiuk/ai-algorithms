import torch
from perceptron import MLP
from manage_data import game_state_to_data_sample
from snake import Direction

class MLPAgent:
    def __init__(self, block_size, bounds, model_path='best_model_weights.pth'):
        self.block_size = block_size
        self.bounds = bounds
        self.model = MLP(16, 32, 4)  # 16 input features, 64 hidden units, 4 output classes
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode

    def act(self, game_state) -> Direction:
        # Process the game state to get the model input
        model_input = game_state_to_data_sample(game_state, self.bounds, self.block_size)
        print(game_state)
        print(model_input)
        model_input = torch.tensor(model_input, dtype=torch.float32)

        # Predict the next move
        with torch.no_grad():
            output = self.model(model_input)
            # testing
            print(f"output: {output}")
            _, predicted = torch.max(output, 1)
            predicted_direction = predicted.item()

        # Map the predicted class to the corresponding direction
        direction_map = {
            0: Direction.UP,
            1: Direction.RIGHT,
            2: Direction.DOWN,
            3: Direction.LEFT,
        }

        action = direction_map[predicted_direction]
        # testing
        print(f"predicted_direction: {action}")

        return action

    def dump_data(self):
        pass
