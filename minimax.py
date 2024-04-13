"""
WSI PROJECT 3

This program implements the MinMaxAgent in a 2-player, turn-based game
using the minimax algorithm.
The two players choose numbers from either the begining or the end
of a given vector in each turn. Their goal is to collect as many
points as possible (equal to the sum of chosen numbers).

The RandomAgent, GreedyAgent, NinjaAgent and run_game implementations
were given. The task was to complete MinMaxAgent and main function.
"""
import random
import time
import statistics as stat
import matplotlib.pyplot as plt

random.seed(15)


class RandomAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if random.random() > 0.5:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class GreedyAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if vector[0] > vector[-1]:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class NinjaAgent:
    """   ⠀⠀⠀⠀⠀⣀⣀⣠⣤⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠴⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠠⠶⠶⠶⠶⢶⣶⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀
⠀⠀⠀⠀⢀⣴⣶⣶⣶⣶⣶⣶⣦⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
⠀⠀⠀⠀⣸⣿⡿⠟⠛⠛⠋⠉⠉⠉⠁⠀⠀⠀⠈⠉⠉⠉⠙⠛⠛⠿⣿⣿⡄⠀
⠀⠀⠀⠀⣿⠋⠀⠀⠀⠐⢶⣶⣶⠆⠀⠀⠀⠀⠀⢶⣶⣶⠖⠂⠀⠀⠈⢻⡇⠀
⠀⠀⠀⠀⢹⣦⡀⠀⠀⠀⠀⠉⢁⣠⣤⣶⣶⣶⣤⣄⣀⠀⠀⠀⠀⠀⣀⣾⠃⠀
⠀⠀⠀⠀⠘⣿⣿⣿⣶⣶⣶⣾⣿⣿⣿⡿⠿⠿⣿⣿⣿⣿⣷⣶⣾⣿⣿⡿⠀⠀
⠀⠀⢀⣴⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀
⠀⠀⣾⡿⢃⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀
⠀⢸⠏⠀⣿⡇⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀
⠀⠀⠀⢰⣿⠃⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⣉⣁⣤⡶⠁⠀⠀⠀⠀⠀
⠀⠀⣠⠟⠁⠀⠀⠀⠀⠀⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀
                かかって来い! """
    def __init__ (OOOO000O000O00000 ):
        OOOO000O000O00000 .numbers =[]
    def act (O000000O000OO0O0O ,O0OO0O0O0O0OO0O00 ):
        if len (O0OO0O0O0O0OO0O00 )%2 ==0 :
            O00O0O0000000OO0O =sum (O0OO0O0O0O0OO0O00 [::2 ])
            O0O00O0OO00O0O0O0 =sum (O0OO0O0O0O0OO0O00 )-O00O0O0000000OO0O
            if O00O0O0000000OO0O >=O0O00O0OO00O0O0O0 :
                O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [0 ])
                return O0OO0O0O0O0OO0O00 [1 :] # explained: https://r.mtdv.me/articles/k1evNIASMp
            O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [-1 ])
            return O0OO0O0O0O0OO0O00 [:-1 ]
        else :
            O00O0O0000000OO0O =max (sum (O0OO0O0O0O0OO0O00 [1 ::2 ]),sum (O0OO0O0O0O0OO0O00 [2 ::2 ]))
            O0O00O0OO00O0O0O0 =max (sum (O0OO0O0O0O0OO0O00 [:-1 :2 ]),sum (O0OO0O0O0O0OO0O00 [:-2 :2 ]))
            if O00O0O0000000OO0O >=O0O00O0OO00O0O0O0 :
                O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [-1 ])
                return O0OO0O0O0O0OO0O00 [:-1 ]
            O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [0 ])
            return O0OO0O0O0O0OO0O00 [1 :]


class MinMaxAgent:
    def __init__(self, max_depth=50):
        self.numbers = []
        self.max_depth = max_depth

    def act(self, vector: list):
        max_val, move = self.minimax(vector, True, 0)
        if move == 'start':
            self.numbers.append(vector[0])
            return vector[1:]
        else:
            self.numbers.append(vector[-1])
            return vector[:-1]

    def minimax(self, vector, max_turn, depth):
        # check if max_depth or terminal state reached
        if depth == self.max_depth or not vector:
            return 0, None

        if max_turn:
            max_val = float('-inf')
            best_move = None
            for move in ['start', 'end']:
                if move == 'start':
                    value = vector[0] + self.minimax(vector[1:], False, depth+1)[0]
                else:
                    value = vector[-1] + self.minimax(vector[:-1], False, depth+1)[0]
                if value > max_val:
                    max_val = value
                    best_move = move
            return max_val, best_move
        else:
            min_val = float('inf')
            for move in ['start', 'end']:
                if move == 'start':
                    value = -vector[0] + self.minimax(vector[1:], True, depth+1)[0]
                else:
                    value = -vector[-1] + self.minimax(vector[:-1], True, depth+1)[0]
                if value < min_val:
                    min_val = value
            return min_val, None


def run_game(vector, first_agent, second_agent):
    while len(vector) > 0:
        vector = first_agent.act(vector)
        if len(vector) > 0:
            vector = second_agent.act(vector)


def main():
    game_num = 0
    exec_times = []
    first_agent_sum = []
    second_agent_sum = []

    # first_agent, second_agent = MinMaxAgent(), GreedyAgent()
    first_agent, second_agent = MinMaxAgent(3), NinjaAgent()
    # first_agent, second_agent = MinMaxAgent(), MinMaxAgent(15)

    while game_num < 1000:
        vector = [random.randint(-10, 10) for _ in range(15)]  

        start_time = time.time()
        # every player should begin the game the same amount of times
        if game_num % 2 == 0:
            run_game(vector, first_agent, second_agent)
        else:
            run_game(vector, second_agent, first_agent)
        
        finish_time = time.time()
        # calculate game execution time
        exec_times.append(finish_time - start_time)

        first_agent_sum.append(sum(first_agent.numbers))
        second_agent_sum.append(sum(second_agent.numbers))

        # print(f"First agent: {sum(first_agent.numbers)} Second agent: {sum(second_agent.numbers)}\n"
        #         f"First agent: {first_agent.numbers}\n"
        #         f"Second agent: {second_agent.numbers}")

        # clear results
        first_agent.numbers = []
        second_agent.numbers = []
        game_num += 1

    print("Average game time: " + str(stat.mean(exec_times)) + "s")
    print()
    print("Average MinMaxAgent score: " + str(stat.mean(first_agent_sum)))
    print("MinMaxAgent score - standard deviation: " + str(stat.stdev(first_agent_sum)))
    print()
    print("Average second agent score: " + str(stat.mean(second_agent_sum)))
    print("Second agent score - standard deviation: " + str(stat.stdev(second_agent_sum)))

    # plot histogram for max_depth = 2 and max_depth = 15
    # if first_agent.max_depth in (2,15):
    #     plt.hist(first_agent_sum, bins=20)
    #     plt.xlabel("MinMaxAgent - suma punktów")
    #     plt.show()



if __name__ == "__main__":
    main()
