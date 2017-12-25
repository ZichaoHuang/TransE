# TransE
A TensorFlow implementation of TransE model in [Translating Embeddings for Modeling
Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf)

# Performance on FB15k
| MeanRank(Raw) | MeanRank(Filter) | Hits@10(Raw)(%) | Hits@10(Filter)(%) |
| :-----------: | :--------------: | :-------------: | :----------------: |
| 161 | 56 | 48.9 | 76.0% |

The above results are obtained by evaluation on the test set after 500 epochs of training.

Download the FB15k dataset from [this repo](https://github.com/thunlp/KB2E).
