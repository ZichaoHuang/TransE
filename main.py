from fb15k import DataSet
from model_transe import TransE

import numpy as np
import argparse
from os import path
from tqdm import tqdm
import tensorflow as tf


def run_training(args):
    dataset = DataSet(data_dir=args.data_dir,
                      negative_sampling=args.negative_sampling)
    model = TransE(learning_rate=args.learning_rate,
                   batch_size=args.batch_size,
                   num_epoch=args.num_epoch,
                   margin=args.margin,
                   embedding_dimension=args.embedding_dimension)

    # construct the training graph
    graph_transe_training = tf.Graph()
    with graph_transe_training.as_default():
        print('constructing the training graph...')
        # generate placeholders for the graph
        with tf.variable_scope('id'):
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

        # model inference
        d_positive, d_negative = model.inference(id_triplets_positive, id_triplets_negative,
                                                 dataset.num_entity, dataset.num_relation)
        # model train loss
        loss = model.loss(d_positive, d_negative)
        # model train operation
        train_op = model.train(loss)
        # model evaluation
        eval_op = model.evaluation(id_triplets_validate)
        print('graph constructing finished')

    # open a session and run the training graph
    with tf.Session(graph=graph_transe_training) as sess:
        # saver for writing training checkpoints
        saver = tf.train.Saver()
        checkpoint_path = path.join(dataset.data_dir, 'checkpoint/model')

        # run the initial operation
        print('initializing all variables...')
        sess.run(tf.global_variables_initializer())
        print('all variables initialized')

        num_batch = dataset.num_triplets_train // model.batch_size

        # training
        print('start training...')
        progressbar_epoch = tqdm(total=model.num_epoch, desc='epoch training')
        for epoch in range(model.num_epoch):
            loss_epoch = 0
            progressbar_batch = tqdm(total=num_batch, desc='batch training', leave=False)
            for batch in range(num_batch):
                batch_positive, batch_negative = dataset.next_batch(model.batch_size)
                feed_dict_train = {
                    id_triplets_positive: batch_positive,
                    id_triplets_negative: batch_negative
                }
                # run the graph
                _, loss_batch = sess.run([train_op, loss], feed_dict=feed_dict_train)
                loss_epoch += loss_batch

                # save a checkpoint and evaluate the model periodically
                if (batch + 1) % 1000 == 0 or (batch + 1) == num_batch:
                    # save a checkpoint
                    save_path = saver.save(
                        sess=sess,
                        save_path=checkpoint_path,
                        global_step=batch
                    )
                    print('\nmodel save at path: {}'.format(save_path))

                    # evaluate the model
                    print('evaluating the current model...')
                    progressbar_eval = tqdm(total=dataset.num_triplets_validate, desc='evaluating')
                    rank = 0
                    for triplet_validate in dataset.triplets_validate:
                        feed_dict_eval = {
                            id_triplets_validate: dataset.next_batch_eval(triplet_validate)
                        }
                        # list of dissimilarity, dissimilarity[0] is the dissimilarity of the valid triplet
                        dissimilarity = sess.run([eval_op], feed_dict=feed_dict_eval)
                        # sort the list, get the rank
                        rank += dissimilarity[0].argsort().argmin()

                        progressbar_eval.update()

                    mean_rank = rank / dataset.num_triplets_validate
                    progressbar_eval.close()

                    print('mean rank at epoch {}, batch {}: {}'.format(epoch, batch, mean_rank))

                # update batch progressbar
                progressbar_batch.set_description(desc='last batch loss: {:.3f}'.format(loss_batch))
                progressbar_batch.update()
            progressbar_batch.close()

            # update epoch progressbar
            progressbar_epoch.set_description(desc='last epoch loss: {:.3f}'.format(loss_epoch))
            progressbar_epoch.update()
        progressbar_epoch.close()


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='initial learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='mini-batch size for SGD'
    )
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=1000,
        help='number of epochs'
    )
    parser.add_argument(
        '--embedding_dimension',
        type=int,
        default=100,
        help='dimension of entity and relation embeddings'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=1.0,
        help='margin of a golden triplet and a corrupted one'
    )

    args = parser.parse_args()
    run_training(args=args)


if __name__ == '__main__':
    main()
