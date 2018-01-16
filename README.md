# TransE
A TensorFlow implementation of TransE model in [Translating Embeddings for Modeling
Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf)

# Performance
| Datasets | MeanRank(Raw) | MeanRank(Filter) | Hits@10(Raw)(%) | Hits@10(Filter)(%) | Epochs |
| :------: | :-----------: | :--------------: | :-------------: | :----------------: | :----: |
| WN18 | 243 | 231 | 79.9 | 93.9 | 1000 |
| FB15k | 246 | 92 | 47.7 | 74.0 | 2000 |

Download the datasets(WN18 and FB15k) from [this repo](https://github.com/thunlp/KB2E).
