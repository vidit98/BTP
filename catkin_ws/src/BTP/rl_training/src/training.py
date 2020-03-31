#!/usr/bin/env python
import gym
import numpy
import time
import rospy
import qlearn
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

if __name__ == '__main__':
	rospy.init_node('test_RL_PANDA', anonymous=True, log_level=rospy.WARN)

	task_and_robot_environment_name = rospy.get_param('/panda/task_and_robot_environment_name')

	env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)

	rospy.loginfo("Gym environment done")
	rospy.loginfo("Starting Learning")
	rospack = rospkg.RosPack()
	pkg_path = rospack.get_path('rl_training')
	outdir = pkg_path + '/training_results'


	last_time_steps = numpy.ndarray(0)

	Alpha = rospy.get_param("/panda/alpha")
	Epsilon = rospy.get_param("/panda/epsilon")
	Gamma = rospy.get_param("/panda/gamma")
	epsilon_discount = rospy.get_param("/panda/epsilon_discount")
	nepisodes = rospy.get_param("/panda/nepisodes")
	nsteps = rospy.get_param("/panda/nsteps")
	running_step = rospy.get_param("/panda/running_step")


	qlearn = qlearn.QLearn(actions=range(env.action_space.n), alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
	initial_epsilon = qlearn.epsilon

	start_time = time.time()
	highest_reward = 0
	for x in range(nepisodes):
		rospy.logwarn("############### START EPISODE=>" + str(x))
		observation = env.reset()
		cumulated_reward = 0
		done = False
		state = ''.join(map(str, observation))
		for i in range(nsteps):
			#rospy.logwarn("############### Start Step=>" + str(i))
			# Pick an action based on the current state
			action = qlearn.chooseAction(state)
			#rospy.logwarn("Next action is:%d", action)
			# Execute the action in the environment and get feedback
			observation, reward, done, info = env.step(action)
			#rospy.logwarn(str(observation) + " " + str(reward))
			#rospy.logwarn("+++++reward" + str(reward))
			cumulated_reward += reward
			if highest_reward < cumulated_reward:
				highest_reward = cumulated_reward
				nextState = ''.join(map(str, observation))
				# Make the algorithm learn based on the results
				rospy.logwarn("# state we were=>" + str(state))
				rospy.logwarn("# action that we took=>" + str(action))
				rospy.logwarn("# reward that action gave=>" + str(reward))
				rospy.logwarn("# episode cumulated_reward=>" + str(cumulated_reward))
				rospy.logwarn("# State in which we will start next step=>" + str(nextState))
				qlearn.learn(state, action, reward, nextState)
				if not (done):
					rospy.logwarn("NOT DONE")
					state = nextState
				else:
					rospy.logwarn("DONE")
					last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
					break
				rospy.logwarn("############### END Step=>" + str(i))
				#raw_input("Next Step...PRESS KEY")
				# rospy.sleep(2.0)
		m, s = divmod(int(time.time() - start_time), 60)
		h, m = divmod(m, 60)
		rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
			round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
			cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

	rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
		initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

	l = last_time_steps.tolist()
	l.sort()
	print last_time_steps
	# print("Parameters: a="+str)
	rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
	rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
	env.close()