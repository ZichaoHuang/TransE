import math

import tensorflow as tf


class TransE:
    def __init__(self,
                 learning_rate,
                 batch_size,
                 num_epoch,
                 embedding_dimension,
                 margin):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.embedding_dimension = embedding_dimension
        self.margin = margin

    def inference(self, dataset, id_triplet_positive, id_triplet_negative):
        minval = -6 / math.sqrt(self.embedding_dimension)
        maxval = 6 / math.sqrt(self.embedding_dimension)
        with tf.variable_scope('embedding'):
            embedding_entity = tf.get_variable(
                name='entity',
                initializer=tf.random_uniform(
                    shape=[dataset.num_entity, self.embedding_dimension],
                    minval=minval,
                    maxval=maxval
                )
            )
            embedding_relation = tf.get_variable(
                name='relation',
                initializer=tf.random_uniform(
                    shape=[dataset.num_relation, self.embedding_dimension],
                    minval=minval,
                    maxval=maxval
                )
            )
            # l2 normalization for relation vector during initialization
            embedding_relation = tf.clip_by_norm(embedding_relation, clip_norm=1, axes=1)

        # embedding lookup, normalize entity embeddings but not relation ones
        embedding_head_positive = tf.nn.embedding_lookup(embedding_entity, id_triplet_positive[:, 0], max_norm=1)
        embedding_head_negative = tf.nn.embedding_lookup(embedding_entity, id_triplet_negative[:, 0], max_norm=1)
        embedding_relation = tf.nn.embedding_lookup(embedding_relation, id_triplet_positive[:, 1])
        embedding_tail_positive = tf.nn.embedding_lookup(embedding_entity, id_triplet_positive[:, 2], max_norm=1)
        embedding_tail_negative = tf.nn.embedding_lookup(embedding_entity, id_triplet_negative[:, 2], max_norm=1)

        # dissimilarity calculation, using euclidean distance
        d_positive = tf.sqrt(tf.reduce_sum(tf.square(embedding_head_positive + embedding_relation
                                                     - embedding_tail_positive), axis=1))
        d_negative = tf.sqrt(tf.reduce_sum(tf.square(embedding_head_negative + embedding_relation
                                                     - embedding_tail_negative), axis=1))

        return d_positive, d_negative

    def loss(self, d_positive, d_negative):
        return tf.reduce_sum(tf.nn.relu(tf.constant(self.margin) + d_positive - d_negative))

    def train(self, loss):
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)  # loss drop really fast by using this
        train_op = optimizer.minimize(loss)

        return train_op
