#!/usr/bin/python

import os
import time

import tensorflow as tf
import numpy as np

from util.system import setup_environment
from model import model_p
from util.app_config import config as app_config
# from util.train import get_trainable_variables, get_learning_rate
from util.losses import regularization_loss
from util.dataloader import get_data

import open3d as o3d


def pc_normalize(pc):
	centroid = np.mean(pc, axis=0)
	pc = pc - centroid
	m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
	pc = pc / m
	return pc





def mkdir_if_missing(path):
	if not os.path.exists(path):
		os.makedirs(path)

pc_data = get_data("/home/vidit/implementation/BTP/data/probe/pcl/")
train_ds = tf.data.Dataset.from_tensor_slices(
     (pc_data)).shuffle(10000).batch(1)
base = o3d.io.read_point_cloud("/home/vidit/implementation/BTP/data/probe/pcl/ptcld83.pcd")
print(np.array(base.points).shape)
base_pc = tf.convert_to_tensor(pc_normalize(np.array(base.points, dtype=np.float32)))


tfsum = tf.summary


def train():
	cfg = app_config

	setup_environment(cfg)

	train_dir = cfg.checkpoint_dir
	mkdir_if_missing(train_dir)

	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
	# dataset = tf.data.Dataset.from_tensor_slices(data)
	# dataset = dataset.batch(1)
	# iterator = dataset.make_one_shot_iterator()
	# train_data = iterator.get_next()

	# print("SHape2", train_data.get_shape())
	summary_writer = tfsum.create_file_writer(train_dir, flush_millis=10000)
	global_step = tf.compat.v1.train.get_or_create_global_step()
	model_fn = model_p.ModelPointCloud(cfg, global_step)

	
	optimizer = tf.keras.optimizers.Adam(0.01)
	
	
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_scopes = ['transform']
	# var_list = get_trainable_variables(train_scopes)	


	
	def train_step(data, ref):
		with tf.GradientTape() as tape:
			outputs = model_fn(data, ref)
			task_loss = model_fn.get_loss(outputs)
			reg_loss = regularization_loss(train_scopes, cfg)
			loss = task_loss + reg_loss
		gradients = tape.gradient(loss, model_fn.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model_fn.trainable_variables))
		train_loss(loss)

	global_step_val = 0
	while global_step_val < cfg.max_number_of_steps:

		for images in train_ds:
		    train_step(images, base_pc)


		print('step: {0}'
				'loss = {loss_val:.4f}'.format(global_step_val, loss_val=train_loss.result()))


train()

