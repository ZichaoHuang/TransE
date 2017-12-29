# TransE
A TensorFlow implementation of TransE model in [Translating Embeddings for Modeling
Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf)

# Performance
| Datasets | MeanRank(Raw) | MeanRank(Filter) | Hits@10(Raw)(%) | Hits@10(Filter)(%) | Epochs |
| :------: | :-----------: | :--------------: | :-------------: | :----------------: | :----: |
| WN18 | 246 | 234 | 79.6 | 93.5 | 800 |
| FB15k | 156 | 51 | 51.1 | 81.3 | 1000 |

Download the datasets(WN18 and FB15k) from [this repo](https://github.com/thunlp/KB2E).
