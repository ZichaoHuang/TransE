from data_preprocess import DataSet
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
                   embedding_dimension=args.embedding_dimension,
                   num_entity=dataset.num_entity,
                   num_relation=dataset.num_relation)

    # construct the training graph
    graph_transe_training = tf.Graph()
    with graph_transe_training.as_default():
        # generate placeholders for the graph
        with tf.variable_scope('id'):
            id_triplet_positive = tf.placeholder(
                dtype=tf.int32,
                shape=[model.batch_size, 3],
                name='id_triplet_positive'
            )
            id_triplet_negative = tf.placeholder(
                dtype=tf.int32,
                shape=[model.batch_size, 3],
                name='id_triplet_negative'
            )

        # model inference
        d_positive, d_negative = model.inference(id_triplet_positive, id_triplet_negative)
        # model train loss
        loss = model.loss(d_positive, d_negative)
        # model train operation
        train_op = model.train(loss)
        # model evaluation
        # evaluation = model.evaluation()

    # # check initial embedding
    # with tf.Session(graph=graph_transe_training) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     batch_positive, batch_negative = dataset.next_batch(model.batch_size)
    #     feed_dict = {
    #         id_triplet_positive: batch_positive,
    #         id_triplet_negative: batch_negative
    #     }
    #     entity, relation = sess.run([embedding_head_pos, embedding_relation], feed_dict=feed_dict)
    #     print('entity norm: ')
    #     print(np.linalg.norm(entity, ord=2, axis=1))
    #     print('relation norm: ')
    #     print(np.linalg.norm(relation, ord=2, axis=1))

    # open a session and run the training graph
    with tf.Session(graph=graph_transe_training) as sess:
        # saver for writing training checkpoints
        saver = tf.train.Saver()
        checkpoint_path = path.join(dataset.data_dir, 'checkpoint/model.ckpt')

        # run the initial operation
        print('initializing all variables...')
        sess.run(tf.global_variables_initializer())
        print('all variables initialized...')

        num_batch = dataset.num_triplets_train // model.batch_size

        # training
        print('start training...')
        progressbar_epoch = tqdm(total=model.num_epoch, desc='epoch training')
        for epoch in range(model.num_epoch):
            loss_epoch = 0
            progressbar_batch = tqdm(total=num_batch, desc='batch training', leave=False)
            for batch in range(num_batch):
                batch_positive, batch_negative = dataset.next_batch(model.batch_size)
                feed_dict = {
                    id_triplet_positive: batch_positive,
                    id_triplet_negative: batch_negative
                }
                # run the graph
                _, loss_batch = sess.run([train_op, loss], feed_dict=feed_dict)
                loss_epoch += loss_batch

                # save a checkpoint and evaluate the model periodically
                if (batch + 1) % 1000 == 0 or (batch + 1) == num_batch:

                    save_path = saver.save(
                        sess=sess,
                        save_path=checkpoint_path,
                        global_step=batch
                    )
                    print('model save at path: {}'.format(save_path))

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
