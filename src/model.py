import math
import timeit
import random
import numpy as np
import tensorflow as tf
import multiprocessing as mp


class TransE:
    def __init__(self, dataset, embedding_dim, margin_value, score_func,
                 batch_size, eval_batch_size, learning_rate, n_generator, n_rank_calculator):
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator

        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None

        '''ops for evaluation'''
        self.head_prediction_raw = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.tail_prediction_raw = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.head_prediction_filter = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.tail_prediction_filter = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.score_eval = None

        bound = 6 / math.sqrt(self.embedding_dim)

        '''embeddings'''
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[dataset.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[dataset.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)

        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.score_eval = self.evaluate(self.head_prediction_raw, self.tail_prediction_raw,
                                            self.head_prediction_filter, self.tail_prediction_filter)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = head_pos + relation_pos - tail_pos
            distance_neg = head_neg + relation_neg - tail_neg

        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.sqrt(tf.reduce_sum(tf.square(distance_pos), axis=1))
                score_neg = tf.sqrt(tf.reduce_sum(tf.square(distance_neg), axis=1))
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')

        return loss

    def evaluate(self, head_prediction_raw, tail_prediction_raw, head_prediction_filter, tail_prediction_filter):
        with tf.name_scope('lookup'):
            '''Raw'''
            head_prediction_raw_h = tf.nn.embedding_lookup(self.entity_embedding, head_prediction_raw[:, 0])
            head_prediction_raw_t = tf.nn.embedding_lookup(self.entity_embedding, head_prediction_raw[:, 1])
            head_prediction_raw_r = tf.nn.embedding_lookup(self.relation_embedding, head_prediction_raw[:, 2])
            tail_prediction_raw_h = tf.nn.embedding_lookup(self.entity_embedding, tail_prediction_raw[:, 0])
            tail_prediction_raw_t = tf.nn.embedding_lookup(self.entity_embedding, tail_prediction_raw[:, 1])
            tail_prediction_raw_r = tf.nn.embedding_lookup(self.relation_embedding, tail_prediction_raw[:, 2])
            '''Filter'''
            head_prediction_filter_h = tf.nn.embedding_lookup(self.entity_embedding, head_prediction_filter[:, 0])
            head_prediction_filter_t = tf.nn.embedding_lookup(self.entity_embedding, head_prediction_filter[:, 1])
            head_prediction_filter_r = tf.nn.embedding_lookup(self.relation_embedding, head_prediction_filter[:, 2])
            tail_prediction_filter_h = tf.nn.embedding_lookup(self.entity_embedding, tail_prediction_filter[:, 0])
            tail_prediction_filter_t = tf.nn.embedding_lookup(self.entity_embedding, tail_prediction_filter[:, 1])
            tail_prediction_filter_r = tf.nn.embedding_lookup(self.relation_embedding, tail_prediction_filter[:, 2])
        with tf.name_scope('link'):
            distance_head_prediction_raw = head_prediction_raw_h + head_prediction_raw_r - head_prediction_raw_t
            distance_tail_prediction_raw = tail_prediction_raw_h + tail_prediction_raw_r - tail_prediction_raw_t
            distance_head_prediction_filter = head_prediction_filter_h + head_prediction_filter_r - head_prediction_filter_t
            distance_tail_prediction_filter = tail_prediction_filter_h + tail_prediction_filter_r - tail_prediction_filter_t
        with tf.name_scope('score'):
            if self.score_func == 'L1':  # L1 score
                score_head_prediction_raw = tf.reduce_sum(tf.abs(distance_head_prediction_raw), axis=1)
                score_tail_prediction_raw = tf.reduce_sum(tf.abs(distance_tail_prediction_raw), axis=1)
                score_head_prediction_filter = tf.reduce_sum(tf.abs(distance_head_prediction_filter), axis=1)
                score_tail_prediction_filter = tf.reduce_sum(tf.abs(distance_tail_prediction_filter), axis=1)
            else:  # L2 score
                score_head_prediction_raw = tf.sqrt(tf.reduce_sum(tf.square(distance_head_prediction_raw), axis=1))
                score_tail_prediction_raw = tf.sqrt(tf.reduce_sum(tf.square(distance_tail_prediction_raw), axis=1))
                score_head_prediction_filter = tf.sqrt(tf.reduce_sum(tf.square(distance_head_prediction_filter), axis=1))
                score_tail_prediction_filter = tf.sqrt(tf.reduce_sum(tf.square(distance_tail_prediction_filter), axis=1))
        return score_head_prediction_raw, score_tail_prediction_raw, score_head_prediction_filter, score_tail_prediction_filter

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                    'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.dataset.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            batch_loss, _, summary = session.run(fetches=[self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            n_used_triple += len(batch_pos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.3f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.dataset.n_training_triple,
                                                                            batch_loss / len(batch_pos)), end='\r')
        print()
        print('mean loss: {:.3f}'.format(epoch_loss / self.dataset.n_training_triple))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish training-----')
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))

    def generate_training_batch(self, in_queue, out_queue):
        while True:
            raw_batch = in_queue.get()
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []
                corrupt_head_prob = np.random.binomial(1, 0.5)
                for head, tail, relation in batch_pos:
                    head_neg = head
                    tail_neg = tail
                    while True:
                        if corrupt_head_prob:
                            head_neg = random.sample(list(self.dataset.entity_dict.values()), 1)[0]
                        else:
                            tail_neg = random.sample(list(self.dataset.entity_dict.values()), 1)[0]
                        if (head_neg, tail_neg, relation) not in self.dataset.golden_triple_pool:
                            break
                    batch_neg.append((head_neg, tail_neg, relation))
                out_queue.put((batch_pos, batch_neg))

    def launch_evaluation(self, session):
        raw_eval_batch_queue = mp.Queue()
        eval_batch_queue = mp.Queue(10000)
        eval_batch_and_score_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.generate_evaluation_batch, kwargs={'in_queue': raw_eval_batch_queue,
                                                                      'out_queue': eval_batch_queue}).start()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        n_eval_triple = 0
        for raw_eval_batch in self.dataset.next_raw_eval_batch(self.eval_batch_size):
            raw_eval_batch_queue.put(raw_eval_batch)
            n_eval_triple += len(raw_eval_batch)
        print('#eval triple: {}'.format(n_eval_triple))
        for _ in range(self.n_generator):
            raw_eval_batch_queue.put(None)
        print('-----Constructing evaluation batches-----')
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_batch_and_score_queue,
                                                           'out_queue': rank_result_queue}).start()
        for i in range(n_eval_triple):
            head_prediction_batch_raw, tail_prediction_batch_raw, \
                head_prediction_batch_filter, tail_prediction_batch_filter = eval_batch_queue.get()
            score_eval = session.run(fetches=self.score_eval,
                                     feed_dict={self.head_prediction_raw: head_prediction_batch_raw,
                                                self.tail_prediction_raw: tail_prediction_batch_raw,
                                                self.head_prediction_filter: head_prediction_batch_filter,
                                                self.tail_prediction_filter: tail_prediction_batch_filter})
            head_prediction_score_raw, tail_prediction_score_raw, \
                head_prediction_score_filter, tail_prediction_score_filter = score_eval
            eval_batch_and_score_queue.put((head_prediction_score_raw, tail_prediction_score_raw,
                                            head_prediction_score_filter, tail_prediction_score_filter))
            print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                               i + 1,
                                                               n_eval_triple), end='\r')
        print()
        for _ in range(self.n_rank_calculator):
            eval_batch_and_score_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_batch_and_score_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        head_rank_raw_sum = 0
        head_hits10_raw_sum = 0
        tail_rank_raw_sum = 0
        tail_hits10_raw_sum = 0
        head_rank_filter_sum = 0
        head_hits10_filter_sum = 0
        tail_rank_filter_sum = 0
        tail_hits10_filter_sum = 0
        for _ in range(n_eval_triple):
            head_rank_raw, head_hits10_raw, tail_rank_raw, tail_hits10_raw, \
                head_rank_filter, head_hits10_filter, tail_rank_filter, tail_hits10_filter = rank_result_queue.get()
            head_rank_raw_sum += head_rank_raw
            head_hits10_raw_sum += head_hits10_raw
            tail_rank_raw_sum += tail_rank_raw
            tail_hits10_raw_sum += tail_hits10_raw
            head_rank_filter_sum += head_rank_filter
            tail_rank_filter_sum += tail_rank_filter
            head_hits10_filter_sum += head_hits10_filter
            tail_hits10_filter_sum += tail_hits10_filter
        print('-----Raw-----')
        head_meanrank_raw = head_rank_raw_sum / n_eval_triple
        head_hits10_raw = head_hits10_raw_sum / n_eval_triple
        tail_meanrank_raw = tail_rank_raw_sum / n_eval_triple
        tail_hits10_raw = tail_hits10_raw_sum / n_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        print('------Average------')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
                                                         (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter = head_rank_filter_sum / n_eval_triple
        head_hits10_filter = head_hits10_filter_sum / n_eval_triple
        tail_meanrank_filter = tail_rank_filter_sum / n_eval_triple
        tail_hits10_filter = tail_hits10_filter_sum / n_eval_triple
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                                         (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def generate_evaluation_batch(self, in_queue, out_queue):
        while True:
            raw_eval_batch = in_queue.get()
            if raw_eval_batch is None:
                return
            else:
                for head, tail, relation in raw_eval_batch:
                    current_triple = (head, tail, relation)
                    '''Raw'''
                    head_prediction_batch_raw = [current_triple]
                    head_neg_pool = set(self.dataset.entity_dict.values())
                    head_neg_pool.remove(head)
                    head_prediction_batch_raw.extend([(head_neg, tail, relation) for head_neg in head_neg_pool])
                    tail_prediction_batch_raw = [current_triple]
                    tail_neg_pool = set(self.dataset.entity_dict.values())
                    tail_neg_pool.remove(tail)
                    tail_prediction_batch_raw.extend([(head, tail_neg, relation) for tail_neg in tail_neg_pool])
                    '''Filter'''
                    head_prediction_batch_filter = [current_triple]
                    for triple_neg in head_prediction_batch_raw:
                        if triple_neg not in self.dataset.golden_triple_pool:
                            head_prediction_batch_filter.append(triple_neg)
                    tail_prediction_batch_filter = [current_triple]
                    for triple_neg in tail_prediction_batch_raw:
                        if triple_neg not in self.dataset.golden_triple_pool:
                            tail_prediction_batch_filter.append(triple_neg)
                    out_queue.put((head_prediction_batch_raw, tail_prediction_batch_raw,
                                   head_prediction_batch_filter, tail_prediction_batch_filter))

    def calculate_rank(self, in_queue, out_queue):
        while True:
            eval_batch_and_score = in_queue.get()
            if eval_batch_and_score is None:
                in_queue.task_done()
                return
            else:
                head_prediction_score_raw, tail_prediction_score_raw, \
                    head_prediction_score_filter, tail_prediction_score_filter = eval_batch_and_score
                '''Raw'''
                head_rank_raw = np.argsort(head_prediction_score_raw).argmin()
                head_hits10_raw = 1 if head_rank_raw < 10 else 0
                tail_rank_raw = np.argsort(tail_prediction_score_raw).argmin()
                tail_hits10_raw = 1 if tail_rank_raw < 10 else 0
                '''Filter'''
                head_rank_filter = np.argsort(head_prediction_score_filter).argmin()
                head_hits10_filter = 1 if head_rank_filter < 10 else 0
                tail_rank_filter = np.argsort(tail_prediction_score_filter).argmin()
                tail_hits10_filter = 1 if tail_rank_filter < 10 else 0
                out_queue.put((head_rank_raw, head_hits10_raw, tail_rank_raw, tail_hits10_raw,
                               head_rank_filter, head_hits10_filter, tail_rank_filter, tail_hits10_filter))
                in_queue.task_done()
