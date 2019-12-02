
import numpy as np
import scipy.io
import tensorflow as tf
import importlib

from util.point_cloud import pc_point_dropout, pointcloud_voxelize, pc_perspective_transform
from util.gauss_kernel import gauss_smoothen_image, smoothing_kernel
from util.quaternion import \
	quaternion_multiply as q_mul,\
	quaternion_normalise as q_norm,\
	quaternion_rotate as q_rotate,\
	quaternion_conjugate as q_conj

from nets import transform_net
from tensorflow.keras import Model

def tf_repeat_0(input, num):
	orig_shape = input.shape
	e = tf.expand_dims(input, axis=1)
	tiler = [1 for _ in range(len(orig_shape)+1)]
	tiler[1] = num
	tiled = tf.tile(e, tiler)
	new_shape = [-1]
	new_shape.extend(orig_shape[1:])
	final = tf.reshape(tiled, new_shape)
	return final

def get_network(name):
	m = importlib.import_module("nets.{}".format(name))
	return m

def get_dropout_prob(cfg, global_step):
	if not cfg.pc_point_dropout_scheduled:
		return cfg.pc_point_dropout

	exp_schedule = cfg.pc_point_dropout_exponential_schedule
	num_steps = cfg.max_number_of_steps
	keep_prob_start = cfg.pc_point_dropout
	keep_prob_end = 1.0
	start_step = cfg.pc_point_dropout_start_step
	end_step = cfg.pc_point_dropout_end_step
	global_step = tf.cast(global_step, dtype=tf.float32)
	x = global_step / num_steps
	k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
	b = keep_prob_start - k * start_step
	if exp_schedule:
		alpha = tf.math.log(keep_prob_end / keep_prob_start)
		keep_prob = keep_prob_start * tf.exp(alpha * x)
	else:
		keep_prob = k * x + b
	keep_prob = tf.clip_by_value(keep_prob, keep_prob_start, keep_prob_end)
	keep_prob = tf.reshape(keep_prob, [])
	return tf.cast(keep_prob, tf.float32)

def get_smooth_sigma(cfg, global_step):
	num_steps = cfg.max_number_of_steps
	diff = (cfg.pc_relative_sigma_end - cfg.pc_relative_sigma)
	sigma_rel = cfg.pc_relative_sigma + global_step / num_steps * diff
	sigma_rel = tf.cast(sigma_rel, tf.float32)
	return sigma_rel


class ModelPointCloud(Model):	

	def __init__(self, cfg,  global_step=0, bn_decay=None):
		super(ModelPointCloud, self).__init__()
		self._params = cfg
		self._gauss_sigma = None
		self._gauss_kernel = None
		self._sigma_rel = None
		self._global_step = global_step
		self.bn_decay = bn_decay
		self.ident = tf.Variable([1.0, 0.0, 0.0, 0.0], dtype=tf.float32)
		self.setup_sigma()
		self.setup_misc()
		self.loss = tf.keras.losses.MeanSquaredError()
		self.transform_fn = None
		self.out = None

	def cfg(self):
		return self._params

	def setup_sigma(self):
		cfg = self.cfg()
		sigma_rel = get_smooth_sigma(cfg, self._global_step)
		print("SIGMA!!", sigma_rel)
		tf.compat.v2.summary.scalar(name="meta/gauss_sigma_rel", data=sigma_rel, step=tf.compat.v1.train.get_or_create_global_step())
		self._sigma_rel = sigma_rel
		self._gauss_sigma = sigma_rel / cfg.vox_size
		self._gauss_kernel = smoothing_kernel(cfg, sigma_rel)

	def gauss_sigma(self):
		return self._gauss_sigma

	def gauss_kernel(self):
		return self._gauss_kernel

	def setup_misc(self):
		if self.cfg().pose_student_align_loss:
			num_points = 2000
			sigma = 1.0
			values = np.random.normal(loc=0.0, scale=sigma, size=(num_points, 3))
			values = np.clip(values, -3*sigma, +3*sigma)
			self._pc_for_alignloss = tf.Variable(values, name="point_cloud_for_align_loss",
												 dtype=tf.float32)


	def model_predict(self, pcd, is_training=False, reuse=False, predict_for_all=False, alignment=None):
		cfg = self._params

		self.transform_fn = get_network(cfg.encoder_name).transform_netV1(is_training)
		with tf.compat.v1.variable_scope('transform', reuse=reuse):

			self.out = self.transform_fn(pcd)


		return self.out

	def get_dropout_keep_prob(self):
		cfg = self.cfg()
		return get_dropout_prob(cfg, self._global_step)

	def voxelise(self, inputs, outputs, is_training):	
		cfg = self.cfg()
		
		voxel = pointcloud_voxelize(cfg, inputs, outputs, self.gauss_sigma())
		return voxel

	def call(self, inputs, ref, is_training=True, reuse=False):
		cfg = self._params
		ident = self.ident
		output = dict()
		ref1 = tf.expand_dims(ref, axis=0)
		inputs1 = tf.concat([ref1, inputs], axis=1)

		quat = self.model_predict(inputs1, is_training, reuse)
		
		ident = tf.expand_dims(ident, axis=0)
		rotated = self.voxelise(inputs, quat, is_training)
		inp_vox = self.voxelise(ref1,ident, is_training)
		vis_rot = pc_perspective_transform(cfg, inputs, quat)
		vis_base = pc_perspective_transform(cfg, ref1, ident)

		output["quat"] = quat
		output["base"] = inp_vox
		output["rotated"] = rotated
		output["vis_rot"] = vis_rot
		output["vis_base"] = vis_base


		return output


	def get_loss(self,outputs, add_summary=True):

		cfg = self.cfg()

		
		l = 	self.loss(outputs["base"], outputs["rotated"])
		
		return l




