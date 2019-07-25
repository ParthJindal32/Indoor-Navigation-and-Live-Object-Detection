import numpy as np 
import random
import pandas

# r matrix
# read the matrix form the appropriate folder
d = pandas.read_csv("C:\\R tables\\0.csv",header = None,index_col=None)
R = np.asarray(d)

# Q Matrix
Q = np.matrix(np.zeros([11,11]))

# gamma (learning parameter)
gamma = 0.8

# Intiial stage. (Usually to be choosen at random)
avilable_states = [0,1,2,3,4,5,6,7,8,9,10]
initial_state = random.choice(fluffy)




# This function returns all avilable actions in the state given as argument
def avilable_actions(state):
	current_state_row = R[state,]
	av_act = np.where(current_state_row >= 0)[0]
	return av_act



# Get avilable actions in the current state
avilable_act = avilable_actions(initial_state)


# This function chooses at random which action to be performed within the range
# of all the avilable actions
def sample_next_action(avilable_actions_range):
	next_action = int(np.random.choice(avilable_act, 1))
	return next_action


# Sample next action to be performed
action = sample_next_action(avilable_act)


# This function updates the Q matrix according to the path selected and the Q 
# learning algorithm

def update(current_state, action, gamma):
	max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

	if max_index.shape[0] > 1:
		max_index = int(np.random.choice(max_index, size=1))
	else:
		max_index = int(max_index)
	max_value = Q[action, max_index]
	
	# Q learning formula
	Q[current_state, action] = R[current_state, action]	+ gamma * max_value
			

# Update Q matrix
update(initial_state, action, gamma)

#_____________________________________________________________________________________
# Training

# Train over 10,000 iterations. (Re-iterate the process above).
for i in range(10000):
	current_state = np.random.randint(0, int(Q.shape[0]))
	avilable_act = avilable_actions(current_state)
	action = sample_next_action(avilable_act)
	update(current_state, action, gamma)

# Normailze the "Trained" Q matrix
print("Trained Q Matrix: ")
Q = Q / np.max(Q) * 100
print(Q)

#_____________________________________________________________________________________
# Testing


# Goal State = 0
# Best sequence path strting from 2 -> 2,1,0

current_state = 0
steps = [current_state]
# cahnge the value depending upon the R matrix you are using
#for instance for the 0.csv r matrix value in while will be 0 and 1 for 1.csv and so on...
while current_state != 0:

	next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

	if next_step_index.shape[0] > 1:
		next_step_index = int(np.random.choice(next_step_index, size=1))
	else:
		next_step_index = int(next_step_index)

	steps.append(next_step_index)
	current_state = next_step_index


# Print	selected sequence of steps
print("Selected path : ")
print(steps)
# save the Q matrix generated corrosponding to R matrix as a .csv file
pandas.DataFrame(Q).to_csv("C:\\Q table\\Qtable(9).csv",header = None,index = None)
