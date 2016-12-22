from fb15k import DataSet
from model_transe import TransE

import random
import argparse
import time
import math
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
                   evaluate_size=args.evaluate_size)

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
            id_triplets_predict_head = tf.placeholder(
                dtype=tf.int32,
                shape=[dataset.num_entity, 3],
                name='triplets_predict_head'
            )
            id_triplets_predict_tail = tf.placeholder(
                dtype=tf.int32,
                shape=[dataset.num_entity, 3],
                name='triplets_predict_tail'
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
            normalize_relation_op = embedding_relation.assign(tf.clip_by_norm(embedding_relation, clip_norm=1, axes=1))
            normalize_entity_op = embedding_entity.assign(tf.clip_by_norm(embedding_entity, clip_norm=1, axes=1))

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
            predict_head, predict_tail = model.evaluation(id_triplets_predict_head, id_triplets_predict_tail)
        print('graph constructing finished')

        # initialize op
        init_op = tf.global_variables_initializer()
        # merge all the summaries
        merge_summary_op = tf.merge_all_summaries()

    # open a session and run the training graph
    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph_transe_training, config=session_config) as sess:
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
            start_train = time.time()
            for batch in range(num_batch):
                # normalize entity embeddings before every batch
                sess.run(normalize_entity_op)

                batch_positive, batch_negative = dataset.next_batch_train(model.batch_size)
                feed_dict_train = {
                    id_triplets_positive: batch_positive,
                    id_triplets_negative: batch_negative
                }

                # run the optimize op, loss op and summary op
                _, loss_batch, summary = sess.run([train_op, loss, merge_summary_op], feed_dict=feed_dict_train)
                loss_epoch += loss_batch

                # write tensorboard logs
                summary_writer.add_summary(summary, global_step=epoch * num_batch + batch)

                # print an overview and save a checkpoint periodically
                if (batch + 1) % 10 == 0 or (batch + 1) == num_batch:
                    print('epoch {}, batch {}, loss: {}'.format(epoch, batch, loss_batch))

                    # TODO: save a check point

            end_train = time.time()
            print('epoch {}, mean batch loss: {:.3f}, time elapsed last epoch: {:.3f}s'.format(
                epoch,
                loss_epoch / args.num_batch,
                end_train - start_train
            ))

            if (epoch + 1) % 5 == 0:
                # evaluate the model
                run_evaluation(sess,
                               predict_head,
                               predict_tail,
                               model,
                               dataset,
                               id_triplets_predict_head,
                               id_triplets_predict_tail)

        end_total = time.time()
        print('total time elapsed: {:.3f}s'.format(end_total - start_total))
        print('training finished')


def run_evaluation(sess,
                   predict_head,
                   predict_tail,
                   model,
                   dataset,
                   id_triplets_predict_head,
                   id_triplets_predict_tail):
    print('evaluating the current model...')
    start_eval = time.time()
    rank_head = 0
    rank_tail = 0
    hit10_head = 0
    hit10_tail = 0
    for triplet in random.sample(dataset.triplets_validate, model.evaluate_size):
        batch_predict_head, batch_predict_tail = dataset.next_batch_eval(triplet)
        feed_dict_eval = {
            id_triplets_predict_head: batch_predict_head,
            id_triplets_predict_tail: batch_predict_tail
        }
        # rank list of head and tail prediction
        prediction_head, prediction_tail = sess.run([predict_head, predict_tail], feed_dict=feed_dict_eval)

        rank_head_current = prediction_head.argsort().argmin()
        rank_head += rank_head_current
        if rank_head_current < 10:
            hit10_head += 1

        rank_tail_current = prediction_tail.argsort().argmin()
        rank_tail += rank_tail_current
        if rank_tail_current < 10:
            hit10_tail += 1

    rank_head_mean = rank_head // model.evaluate_size
    hit10_head /= model.evaluate_size
    rank_tail_mean = rank_tail // model.evaluate_size
    hit10_tail /= model.evaluate_size
    end_eval = time.time()
    print('head prediction mean rank: {:d}, hit@10: {:.3f}%'.format(rank_head_mean, hit10_head * 100))
    print('tail prediction mean rank: {:d}, hit@10: {:.3f}%'.format(rank_tail_mean, hit10_tail * 100))
    print('time elapsed last evaluation: {:.3f}s'.format(end_eval - start_eval))
    print('back to training...')


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
        default='bern',
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
        default=4800,
        help='mini batch size for SGD'
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
        '--evaluate_size',
        type=int,
        default=500,
        help='the size of evaluate triplets, max is 50000'
    )

    # tensorboard args
    parser.add_argument(
        '--log_dir',
        type=str,
        default='log/',
        help='tensorflow log files directory, for tensorboard'
    )

    args = parser.parse_args()
    print('args: {}'.format(args))
    run_training(args=args)


if __name__ == '__main__':
    main()
