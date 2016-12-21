from fb15k import DataSet
from model_transe import TransE

import random
import argparse
import time
import math
from os import path
import numpy as np
import tensorflow as tf


def run_training(args):
    dataset = DataSet(data_dir=args.data_dir,
                      negative_sampling=args.negative_sampling)
    model = TransE(learning_rate=args.learning_rate,
                   batch_size=args.batch_size,
                   num_epoch=args.num_epoch,
                   margin=args.margin,
                   embedding_dimension=args.embedding_dimension,
                   dissimilarity=args.dissimilarity,
                   validate_size=args.validate_size)

    # construct the training graph
    graph_transe_training = tf.Graph()
    with graph_transe_training.as_default():
        print('constructing the training graph...')

        # generate placeholders for the graph
        with tf.variable_scope('input'):
            id_triplets_positive = tf.placeholder(
                dtype=tf.int32,
                shape=[model.batch_size, 3],
                name='triplets_positive'
            )
            id_triplets_negative = tf.placeholder(
                dtype=tf.int32,
                shape=[model.batch_size, 3],
                name='triplets_negative'
            )
            id_triplets_validate = tf.placeholder(
                dtype=tf.int32,
                shape=[2 * dataset.num_entity - 1, 3],  # 2 * (num_entity - 1) + 1
                name='triplet_validate'
            )

        # embedding table
        bound = 6 / math.sqrt(model.embedding_dimension)
        with tf.variable_scope('embedding'):
            embedding_entity = tf.get_variable(
                name='entity',
                initializer=tf.random_uniform(
                    shape=[dataset.num_entity, model.embedding_dimension],
                    minval=-bound,
                    maxval=bound
                )
            )
            embedding_relation = tf.get_variable(
                name='relation',
                initializer=tf.random_uniform(
                    shape=[dataset.num_relation, model.embedding_dimension],
                    minval=-bound,
                    maxval=bound
                )
            )

        with tf.name_scope('normalization'):
            normalize_relation_op = embedding_relation.assign(tf.clip_by_norm(embedding_relation, clip_norm=1, axes=1,
                                                                              name='relation'))
            normalize_entity_op = embedding_entity.assign(tf.clip_by_norm(embedding_entity, clip_norm=1, axes=1,
                                                                          name='entity'))

        # ops into scopes, convenient for TensorBoard's Graph visualization
        with tf.name_scope('inference'):
            # model inference
            d_positive, d_negative = model.inference(id_triplets_positive, id_triplets_negative)
        with tf.name_scope('loss'):
            # model train loss
            loss = model.loss(d_positive, d_negative)
        with tf.name_scope('optimization'):
            # model train operation
            train_op = model.train(loss)
        with tf.name_scope('evaluation'):
            # model evaluation
            eval_op = model.evaluation(id_triplets_validate)
        print('graph constructing finished')

        # initialize op
        init_op = tf.global_variables_initializer()

        merge_summary_op = tf.merge_all_summaries()

    # open a session and run the training graph
    session_config = tf.ConfigProto(log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph_transe_training, config=session_config) as sess:
        # # saver for writing training checkpoints
        # saver = tf.train.Saver()
        # checkpoint_path = path.join(dataset.data_dir, 'checkpoint/model')

        # run the initial operation
        print('initializing all variables...')
        sess.run(init_op)
        print('all variables initialized')

        # normalize relation embeddings after initialization
        sess.run(normalize_relation_op)

        # op to write logs to tensorboard
        summary_writer = tf.train.SummaryWriter(args.log_dir, graph=sess.graph)

        num_batch = dataset.num_triplets_train // model.batch_size

        # training
        print('start training...')
        start_total = time.time()
        for epoch in range(model.num_epoch):
            loss_epoch = 0
            start = time.time()
            for batch in range(num_batch):
                # normalize entity embeddings before every batch
                sess.run(normalize_entity_op)

                batch_positive, batch_negative = dataset.next_batch_train(model.batch_size)
                feed_dict_train = {
                    id_triplets_positive: batch_positive,
                    id_triplets_negative: batch_negative
                }

                # # initial embedding norm check
                # if batch == 0:
                #     # check embedding norm
                #     entity, relation = sess.run([embedding_entity, embedding_relation], feed_dict=feed_dict_train)
                #
                #     print('initial value:')
                #     print('entity norm:')
                #     print(np.linalg.norm(entity, ord=2, axis=1))
                #     print(np.linalg.norm(entity, ord=2, axis=0))
                #     print('relation norm:')
                #     print(np.linalg.norm(relation, ord=2, axis=1))
                #     print(np.linalg.norm(relation, ord=2, axis=0))
                #     print('entity embedding:')
                #     print(entity)
                #     print('relation embeddings:')
                #     print(relation)
                #     print()

                # run the optimize op, loss op and summary op
                _, loss_batch, summary = sess.run([train_op, loss, merge_summary_op], feed_dict=feed_dict_train)
                loss_epoch += loss_batch

                # write tensorboard logs
                summary_writer.add_summary(summary, global_step=epoch * num_batch + batch)

                # # print an overview of training every 100 steps
                # if batch % 100 == 0:
                #     print('epoch {}, batch {}, loss: {}'.format(epoch, batch, loss_batch))

                #     # check embedding norm
                #     entity, relation = sess.run([embedding_entity, embedding_relation], feed_dict=feed_dict_train)
                #
                #     print('entity norm:')
                #     print(np.linalg.norm(entity, ord=2, axis=1))
                #     print(np.linalg.norm(entity, ord=2, axis=0))
                #     print('relation norm:')
                #     print(np.linalg.norm(relation, ord=2, axis=1))
                #     print(np.linalg.norm(relation, ord=2, axis=0))
                #     print('entity embedding:')
                #     print(entity)
                #     print('relation embedding:')
                #     print(relation)
                #     print()

                # print an overview, save a checkpoint and evaluate the model periodically
                if (batch + 1) % 10 == 0 or (batch + 1) == num_batch:
                    print('epoch {}, batch {}, loss: {}'.format(epoch, batch, loss_batch))
                    # # save a checkpoint
                    # save_path = saver.save(
                    #     sess=sess,
                    #     save_path=checkpoint_path,
                    #     global_step=batch
                    # )
                    # print('model save at: {}'.format(save_path))

                    # evaluate the model
                    print('evaluating the current model...')
                    rank = 0
                    for triplet_validate in random.sample(dataset.triplets_validate, model.validate_size):
                        feed_dict_eval = {
                            id_triplets_validate: dataset.next_batch_eval(triplet_validate)
                        }
                        # list of dissimilarity, the first element in the list is the dissimilarity of the valid triplet
                        dissimilarity = sess.run(eval_op, feed_dict=feed_dict_eval)
                        # sort the list, get the rank of dissimilarity[0], which is argmin()
                        rank += dissimilarity.argsort().argmin()
                    mean_rank = int(rank / model.validate_size)
                    print('mean rank: {:d}'.format(mean_rank))
                    print('back to training...')
            end = time.time()
            print('epoch {}, mean batch loss: {:.3f}, time elapsed last epoch: {:.3f}'.format(
                epoch,
                loss_epoch / num_batch,
                end - start
            ))
        end_total = time.time()
        print('total time elapsed: {:.3f}s'.format(end_total - start_total))
        print('training finished')


def main():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset/FB15k',
        help='dataset directory'
    )
    parser.add_argument(
        '--negative_sampling',
        type=str,
        default='unif',
        help='negative sampling method, unif or bern'
    )

    # model args
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='initial learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5000,
        help='mini-batch size for optimization'
    )
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=1000,
        help='number of epochs'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=1.0,
        help='margin of a golden triplet and a corrupted one'
    )
    parser.add_argument(
        '--embedding_dimension',
        type=int,
        default=50,
        help='dimension of entity and relation embeddings'
    )
    parser.add_argument(
        '--dissimilarity',
        type=str,
        default='L1',
        help='using L1 or L2 distance as dissimilarity'
    )
    parser.add_argument(
        '--validate_size',
        type=int,
        default=1000,
        help='the size of validation set, max is 50000'
    )

    # tensorboard args
    parser.add_argument(
        '--log_dir',
        type=str,
        default='log/',
        help='tensorflow log files directory, for tensorboard'
    )

    args = parser.parse_args()
    run_training(args=args)


if __name__ == '__main__':
    main()
