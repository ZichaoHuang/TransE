import math

import tensorflow as tf


class TransE:
    def __init__(self,
                 learning_rate,
                 batch_size,
                 num_epoch,
                 margin,
                 embedding_dimension,
                 dissimilarity,
                 validate_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.margin = margin
        self.embedding_dimension = embedding_dimension
        self.dissimilarity = dissimilarity
        self.validate_size = validate_size

    def inference(self, id_triplets_positive, id_triplets_negative, num_entity, num_relation):
        minval = -6 / math.sqrt(self.embedding_dimension)
        maxval = 6 / math.sqrt(self.embedding_dimension)
        with tf.variable_scope('embedding'):
            embedding_entity = tf.get_variable(
                name='entity',
                initializer=tf.random_uniform(
                    shape=[num_entity, self.embedding_dimension],
                    minval=minval,
                    maxval=maxval
                )
            )
            embedding_relation = tf.get_variable(
                name='relation',
                initializer=tf.random_uniform(
                    shape=[num_relation, self.embedding_dimension],
                    minval=minval,
                    maxval=maxval
                )
            )
            # l2 normalization for relation vector during initialization
            embedding_relation = tf.clip_by_norm(embedding_relation, clip_norm=1, axes=1)

        # embedding lookup, normalize entity embeddings but not relation ones
        embedding_head_positive = tf.nn.embedding_lookup(embedding_entity, id_triplets_positive[:, 0], max_norm=1)
        embedding_head_negative = tf.nn.embedding_lookup(embedding_entity, id_triplets_negative[:, 0], max_norm=1)
        embedding_relation = tf.nn.embedding_lookup(embedding_relation, id_triplets_positive[:, 1])
        embedding_tail_positive = tf.nn.embedding_lookup(embedding_entity, id_triplets_positive[:, 2], max_norm=1)
        embedding_tail_negative = tf.nn.embedding_lookup(embedding_entity, id_triplets_negative[:, 2], max_norm=1)

        if self.dissimilarity == 'L2':
            d_positive = tf.sqrt(tf.reduce_sum(tf.square(embedding_head_positive + embedding_relation
                                                         - embedding_tail_positive), axis=1))
            d_negative = tf.sqrt(tf.reduce_sum(tf.square(embedding_head_negative + embedding_relation
                                                         - embedding_tail_negative), axis=1))
        else:  # default: L1
            d_positive = tf.reduce_sum(tf.abs(embedding_head_positive + embedding_relation
                                              - embedding_tail_positive), axis=1)
            d_negative = tf.reduce_sum(tf.abs(embedding_head_negative + embedding_relation
                                              - embedding_tail_negative), axis=1)

        return d_positive, d_negative

    def loss(self, d_positive, d_negative):
        return tf.reduce_sum(tf.nn.relu(tf.constant(self.margin) + d_positive - d_negative), name='max_margin_loss')

    def train(self, loss):
        # add a scalar summary for the snapshot loss
        tf.scalar_summary(loss.op.name, loss)

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)  # loss drop really fast by using this
        train_op = optimizer.minimize(loss)

        return train_op

    def evaluation(self, id_triplets_validate):
        # get one single validate triplet and do evaluation
        with tf.variable_scope('embedding', reuse=True):  # reusing variables: reuse=True
            embedding_entity = tf.get_variable(name='entity')
            embedding_relation = tf.get_variable(name='relation')

        embedding_head = tf.nn.embedding_lookup(embedding_entity, id_triplets_validate[:, 0])
        embedding_relation = tf.nn.embedding_lookup(embedding_relation, id_triplets_validate[:, 1])
        embedding_tail = tf.nn.embedding_lookup(embedding_entity, id_triplets_validate[:, 2])

        if self.dissimilarity == 'L2':
            dissimilarity = tf.sqrt(tf.reduce_sum(tf.square(embedding_head + embedding_relation
                                                            - embedding_tail), axis=1))
        else:
            dissimilarity = tf.reduce_sum(tf.abs(embedding_head + embedding_relation - embedding_tail), axis=1)

        return dissimilarity
