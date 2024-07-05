from typing import Any, Dict
# import pygame
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from examples.DQN import DEVICE, DQNAgent
import random
import statistics
import torch.optim as optim
import torch 
import tqdm
import datetime
import distutils.util

from game import AI_PLAYER_ID, GameStep, PlayerDecision, get_game_score, initialize_game_state, RandomPlayer, play_game_until_decision_one_player_that_is_not_a_shop_decision, set_decision

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

#################################
#   Define parameters manually  #
#################################
def define_parameters():
	params: Dict[str, Any] = {}
	# Neural Network
	params['epsilon_decay_linear'] = 1/100
	params['learning_rate'] = 0.00013629
	params['first_layer_size'] = 200	# neurons in the first layer
	params['second_layer_size'] = 20   # neurons in the second layer
	params['third_layer_size'] = 50	# neurons in the third layer
	params['episodes'] = 1000		  
	params['memory_size'] = 2500
	params['batch_size'] = 300
	# Settings
	params['weights_path'] = 'weights/weights.h5'
	params['train'] = True
	params["test"] = False
	params['plot_score'] = True
	params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
	return params


# class Game:
#	 """ Initialize PyGAME """
		
#	 def __init__(self, game_width, game_height):
#		 pygame.display.set_caption('SnakeGen')
#		 self.game_width = game_width
#		 self.game_height = game_height
#		 self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
#		 self.bg = pygame.image.load("img/background.png")
#		 self.crash = False
#		 self.player = Player(self)
#		 self.food = Food()
#		 self.score = 0


def display_ui(game, score, record):
	myfont = pygame.font.SysFont('Segoe UI', 20)
	myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
	text_score = myfont.render('SCORE: ', True, (0, 0, 0))
	text_score_number = myfont.render(str(score), True, (0, 0, 0))
	text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
	text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
	game.gameDisplay.blit(text_score, (45, 440))
	game.gameDisplay.blit(text_score_number, (120, 440))
	game.gameDisplay.blit(text_highest, (190, 440))
	game.gameDisplay.blit(text_highest_number, (350, 440))
	game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record):
	game.gameDisplay.fill((255, 255, 255))
	display_ui(game, game.score, record)
	player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
	food.display_food(food.x_food, food.y_food, game)


def initialize_game(player, game, food, agent, batch_size):
	state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
	action = [1, 0, 0]
	player.do_move(action, player.x, player.y, game, food, agent)
	state_init2 = agent.get_state(game, player, food)
	reward1 = agent.set_reward(player, game.crash)
	agent.remember(state_init1, action, reward1, state_init2, game.crash)
	agent.replay_new(agent.memory, batch_size)


def plot_seaborn(array_counter, array_score, train):
	sns.set(color_codes=True, font_scale=1.5)
	sns.set_style("white")
	plt.figure(figsize=(13,8))
	fit_reg = False if train== False else True		
	ax = sns.regplot(
		np.array([array_counter])[0],
		np.array([array_score])[0],
		#color="#36688D",
		x_jitter=.1,
		scatter_kws={"color": "#36688D"},
		label='Data',
		fit_reg = fit_reg,
		line_kws={"color": "#F49F05"}
	)
	# Plot the average line
	y_mean = [np.mean(array_score)]*len(array_counter)
	ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
	ax.legend(loc='upper right')
	ax.set(xlabel='# games', ylabel='score')
	plt.show()


def get_mean_stdev(array):
	return statistics.mean(array), statistics.stdev(array)


def test(params):
	params['load_weights'] = True
	params['train'] = False
	params["test"] = False 
	score, mean, stdev = run(params)
	return score, mean, stdev


def run(params: Dict[str, Any]):

	agent = DQNAgent(params)
	agent = agent.to(DEVICE)
	agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
	counter_games = 0
	score_plot = []
	counter_plot = []
	record = 0
	total_score = 0
	# while counter_games < params['episodes']:

	
	for _ in tqdm.tqdm(range(params['episodes']), desc="Episodes"):

		# print("Episode " + str(counter_games))
		# for event in pygame.event.get():
		# 	if event.type == pygame.QUIT:
		# 		pygame.quit()
		# 		quit()

		# Initialize classes
		
		random_player = RandomPlayer()
		game = initialize_game_state()
		play_game_until_decision_one_player_that_is_not_a_shop_decision(game, random_player)
 
		steps = 0	# steps since the last positive reward

		while (not game.state == GameStep.STATE_END and not game.state == GameStep.STATE_ERROR):
			# print("Game state", game.state)
			play_game_until_decision_one_player_that_is_not_a_shop_decision(game, random_player)

			if not params['train']:
				agent.epsilon = 0.01
			else:
				# agent.epsilon is set to give randomness to actions
				agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

			# get old state
			state_old = agent.get_state(game, AI_PLAYER_ID)

			STATE_TENSOR_SIZE = len(state_old)

			# perform random actions based on agent.epsilon, or choose the action
			if random.uniform(0, 1) < agent.epsilon:
				random_decision = random_player.run_player_decision(game, AI_PLAYER_ID)
				# final_move = np.eye(3)[randint(0,2)]
				final_move = np.array(random_decision.to_state_array())

				reverse = PlayerDecision.from_state_array(final_move).to_state_array()
				if random_decision != PlayerDecision.from_state_array(final_move):
					new_decision = PlayerDecision.from_state_array(final_move)

			else:
				# predict action based on the old state
				with torch.no_grad():
					state_old_tensor = torch.tensor(state_old.reshape((1, STATE_TENSOR_SIZE)), dtype=torch.float32).to(DEVICE)
					prediction = agent(state_old_tensor)
					# print("Prediction", prediction)
					# final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]
					final_move = prediction.detach().cpu().numpy()[0]


			# perform new move and get new state
			if False: # True: # params['display']:
				print("State:", game)
				print("Decision:", PlayerDecision.from_state_array(final_move))

			set_decision(game, PlayerDecision.from_state_array(final_move), AI_PLAYER_ID)
			play_game_until_decision_one_player_that_is_not_a_shop_decision(game, random_player)

			if game.state == GameStep.STATE_ERROR:
				# Replay the last move
				set_decision(game, PlayerDecision.from_state_array(final_move), AI_PLAYER_ID)

			state_new = agent.get_state(game, AI_PLAYER_ID)

			# set reward for the new state
			reward = agent.set_reward(game)
			# print("Reward: ", reward)
			
			# if food is eaten, steps is set to 0
			# if reward > 0:
			# 	steps = 0
				
			if params['train']:
				# train short memory base on the new action and state
				agent.train_short_memory(state_old, final_move, reward, state_new, game.state == GameStep.STATE_ERROR)
				# store the new data into a long term memory
				agent.remember(state_old, final_move, reward, state_new, game.state == GameStep.STATE_ERROR)

			# record = get_record(game.score, record)

			# if params['display']:
			# 	display(player1, food1, game, record)

			steps += 1


		if params['train']:
			agent.replay_new(agent.memory, params['batch_size'])

		counter_games += 1
		total_score += get_game_score(game)
		# print(f'Game {counter_games}	  Reward: {reward}')
		score_plot.append(get_game_score(game))
		counter_plot.append(counter_games)
	mean, stdev = get_mean_stdev(score_plot)
	if params['train']:
		model_weights = agent.state_dict()
		torch.save(model_weights, params["weights_path"])
	if params['plot_score']:
		plot_seaborn(counter_plot, score_plot, params['train'])
	return total_score, mean, stdev

if __name__ == '__main__':
	# Set options to activate or deactivate the game view, and its speed
	# pygame.font.init()
	parser = argparse.ArgumentParser()
	params = define_parameters()
	parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
	parser.add_argument("--speed", nargs='?', type=int, default=50)
	parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)
	args = parser.parse_args()
	print("Args", args)
	params['display'] = args.display
	params['speed'] = args.speed

	if params['train']:
		print("Training...")
		params['load_weights'] = False   # when training, the network is not pre-trained
		run(params)
	if params['test']:
		print("Testing...")
		params['train'] = False
		params['load_weights'] = True
		run(params)