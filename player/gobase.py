import sys
import numpy as np
import tensorflow as tf
import argparse
import os
import time

def resnet_block(x,n_channel,scope=None,reuse=True,regularizer=None):
    with tf.variable_scope(scope):
        shortcut=x
        x=tf.contrib.layers.conv2d(x,num_outputs=n_channel,kernel_size=3,reuse=reuse,scope='conv0',weights_regularizer=regularizer)
        x=tf.contrib.layers.conv2d(x,num_outputs=n_channel,kernel_size=3,activation_fn=None,reuse=reuse,scope='conv1',weights_regularizer=regularizer)
        x=shortcut+x
        x=tf.nn.relu(x)

    return x

class GoModel(object):
    def __init__(self, args):
        self.board_size=9
        self.model_path=args.model_path
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True
        self.graph = tf.Graph()
        self.sess=tf.Session(graph=self.graph, config=config)
        self.cur_checkpoint='None'

    def _forward(self, inputs, reuse=None, regularizer=None):
        x=inputs
        n_filter=64

        net=tf.contrib.layers.conv2d(x,n_filter,kernel_size=3, biases_initializer=None, activation_fn=None, scope="input",reuse=reuse,weights_regularizer=regularizer)
        net=resnet_block(net,n_filter,reuse=reuse,regularizer=regularizer, scope="resnet0")
        net=resnet_block(net,n_filter,reuse=reuse,regularizer=regularizer, scope="resnet1")
        net=resnet_block(net,n_filter,reuse=reuse,regularizer=regularizer, scope="resnet2")
        net=resnet_block(net,n_filter,reuse=reuse,regularizer=regularizer, scope="resnet3")
        net=resnet_block(net,n_filter,reuse=reuse,regularizer=regularizer, scope="resnet4")
        net=resnet_block(net,n_filter,reuse=reuse,regularizer=regularizer, scope="resnet5")

        #value
        value = tf.contrib.layers.conv2d(net,num_outputs=1,kernel_size=1,reuse=reuse,scope='value1',weights_regularizer=regularizer)
        value = tf.contrib.layers.flatten(value)
        value = tf.contrib.layers.fully_connected(value, 128, scope='value2',reuse=reuse,weights_regularizer=regularizer)
        value = tf.contrib.layers.fully_connected(value, 1, scope='value3',reuse=reuse,weights_regularizer=regularizer,activation_fn=None)
        value = tf.squeeze(value,axis=1)

        #policy
        policy = tf.contrib.layers.conv2d(net,num_outputs=2,kernel_size=1,reuse=reuse,scope='policy1',weights_regularizer=regularizer)
        policy = tf.contrib.layers.flatten(policy)
        logit = tf.contrib.layers.fully_connected(policy,82,scope='policy2',reuse=reuse,weights_regularizer=regularizer,activation_fn=None)
        pred=tf.nn.softmax(logit)

        return logit,pred,value


    def _loss(self, logit,value,nextmove,label,regularizer):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=nextmove)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        v_diff = tf.squared_difference(value, label)
        v_loss = tf.reduce_mean(v_diff)

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.add_n(reg_variables)

        return cross_entropy_mean,v_loss,reg_term,v_loss+cross_entropy_mean+reg_term
           
    def build(self):
        with self.graph.as_default():
            self.global_step= tf.Variable(0, name='global_step', trainable=False)
            feature = self.feature = tf.placeholder(tf.float32, shape=[None,9,9,10], name='sample')
            nextmove = self.nextmove = tf.placeholder(tf.float32, shape=[None,82], name='nextmove')
            label = self.label = tf.placeholder(tf.float32, shape=[None], name='label')
            with tf.device('/gpu:0'):
                regularizer=tf.contrib.layers.l2_regularizer(0.0001)

                logit,prob,value = self._forward(self.feature,regularizer=regularizer)
                self.prob = tf.nn.softmax(logit)
                self.value = value

                ce,mse,reg,loss = self._loss(logit,value,self.nextmove,self.label,regularizer=regularizer)
                opt=tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9)
                grads = opt.compute_gradients(loss)
                self.train_step=opt.apply_gradients(grads,self.global_step)
                self.loss=loss

                tf.summary.scalar('mse', mse)
                tf.summary.scalar('ce',ce)
                tf.summary.scalar('reg',reg)
                tf.summary.scalar('loss',loss)
                self.summary_step=tf.summary.merge_all()

            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=5)

    def train(self,feature,nextmove,label,summary_writer=None):
        start_time=time.time()

        feed_dict={self.feature:feature,self.nextmove:nextmove,self.label:label}
        if not summary_writer is None:      
            global_step=self.get_step()
            _, loss,summary = self.sess.run([self.train_step, self.loss, self.summary_step],feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step)
            print 'step=%d loss=%.3f'%(global_step,loss)
        else:
            self.sess.run(self.train_step,feed_dict=feed_dict) 
        global_step=self.get_step() 

    def get_step(self):
        global_step = self.sess.run(self.global_step)
        return global_step

    def predict(self,features):
        feed = {self.feature:features}
        prob,value = self.sess.run([self.prob,self.value],feed_dict=feed)
        return prob,value 

    def save_model(self, path=None):
        if path is None:
            path=self.model_path
        self.saver.save(self.sess, os.path.join(path,'model.ckpt'), global_step=self.global_step)

    def load_model(self, path=None):
        if path is None:
            path=self.model_path
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            if ckpt.model_checkpoint_path==self.cur_checkpoint:
                sys.stderr.write("same model %s, ignore load\n"%(ckpt.model_checkpoint_path))
            else:
                sys.stderr.write("Load model %s\n" % (ckpt.model_checkpoint_path))
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.cur_checkpoint=str(ckpt.model_checkpoint_path)
            return True
        else:
            sys.stderr.write("No model.\n")
            return False

