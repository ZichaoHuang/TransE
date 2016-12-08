import random
import csv

from numpy.random import binomial
from pathlib import Path
from itertools import accumulate


class DataSet:

    def __init__(self, data_dir, negative_sampling='unif'):
        self.data_dir = data_dir
        self.negative_sampling = negative_sampling

        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}

        self.triplets_train_pool = set()  # {(id_head, id_relation, id_tail), ...}
        self.triplets_train = []  # [(id_head, id_relation, id_tail), ...]
        self.triplets_valid = []
        self.triplets_test = []

        self.num_entity = 0
        self.num_relation = 0
        self.num_triplets_train = 0

        # for reducing false negative labels
        self.count_head = {}  # {relation, {head, tail_count}}
        self.count_tail = {}  # {relation, {tail, head_count}}
        self.relation_hpt_tph = {}  # {relation, (head per tail, tail per head)}
        self.load_data()

    def load_data(self):
        # read the entity2id file
        with Path(self.data_dir).joinpath('entity2id.txt').open() as file:
            print('loading entity file...')
            for entity, id_entity in csv.reader(file, delimiter='\t'):
                self.entity2id[entity] = id_entity
                self.id2entity[id_entity] = entity
            self.num_entity = len(self.entity2id)
            print('get {} entities'.format(self.num_entity))

        # read the relation2id file
        with Path(self.data_dir).joinpath('relation2id.txt').open() as file:
            print('loading relation file...')
            for relation, id_relation in csv.reader(file, delimiter='\t'):
                self.relation2id[relation] = id_relation
                self.id2relation[id_relation] = relation
            self.num_relation = len(self.relation2id)
            print('get {} relations'.format(self.num_relation))

        # read the train file
        with Path(self.data_dir).joinpath('train.txt').open() as file:
            print('loading train triplets file...')
            for head, tail, relation in csv.reader(file, delimiter='\t'):
                id_head = self.entity2id[head]
                id_relation = self.relation2id[relation]
                id_tail = self.entity2id[tail]

                self.triplets_train.append((id_head, id_relation, id_tail))

                # for reducing false negative labels
                if self.negative_sampling == 'bern':
                    if id_relation not in self.count_head:  # new relation
                        self.count_head[id_relation] = {id_head: 1}
                    else:
                        if id_head not in self.count_head[id_relation]:  # new head
                            self.count_head[id_relation][id_head] = 1
                        else:
                            self.count_head[id_relation][id_head] += 1

                    if id_relation not in self.count_tail:  # new relation
                        self.count_tail[id_relation] = {id_tail: 1}
                    else:
                        if id_tail not in self.count_tail[id_relation]:  # new tail
                            self.count_tail[id_relation][id_tail] = 1
                        else:
                            self.count_tail[id_relation][id_tail] += 1

            self.num_triplets_train = len(self.triplets_train)
            print('get {} triplets from training set'.format(self.num_triplets_train))

        # TODO: read the valid file

        # TODO: read the test file

        # construct the train triplets pool
        self.triplets_train_pool = set(self.triplets_train)

        if self.negative_sampling == 'bern':
            self.bernoulli_setting()
        else:
            print('do not need to calculate hpt & tph...')

    def bernoulli_setting(self):
        print('calculating hpt & tph for reducing negative false labels...')
        for id_relation in self.relation2id.values():
            hpt = list(accumulate(list(self.count_tail[id_relation].values())))[-1] / len(self.count_tail[id_relation])
            tph = list(accumulate(list(self.count_head[id_relation].values())))[-1] / len(self.count_head[id_relation])
            self.relation_hpt_tph[id_relation] = (hpt, tph)

    def next_batch(self, batch_size):
        # construct positive batch
        batch_positive = random.sample(self.triplets_train, batch_size)

        # construct negative batch
        batch_negative = []
        for id_head, id_relation, id_tail in batch_positive:
            id_head_corrupted = id_head
            id_tail_corrupted = id_tail

            head_prob = binomial(1, 0.5)  # default: unif
            if self.negative_sampling == 'bern':  # bern
                hpt, tph = self.relation_hpt_tph[id_relation]
                head_prob = binomial(1, (tph / (tph + hpt)))

            # corrupt head or tail, but not both
            while True:
                if head_prob:  # replace head
                    id_head_corrupted = random.sample(list(self.entity2id.values()), 1)[0]
                else:  # replace tail
                    id_tail_corrupted = random.sample(list(self.entity2id.values()), 1)[0]

                if (id_head_corrupted, id_relation, id_tail_corrupted) not in self.triplets_train_pool:
                    break
            batch_negative.append((id_head_corrupted, id_relation, id_tail_corrupted))

        return batch_positive, batch_negative
