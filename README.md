# TransE
A TensorFlow implementation of TransE model in [Translating Embeddings for Modeling
Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf)

# Performance
| Datasets | MeanRank(Raw) | MeanRank(Filter) | Hits@10(Raw)(%) | Hits@10(Filter)(%) | Epochs |
| :------: | :-----------: | :--------------: | :-------------: | :----------------: | :----: |
| WN18 | 220 | 208 | 78.4 | 91.3 | 300 |
| FB15k | 161 | 56 | 48.9 | 76.0 | 500 |

Download the datasets(WN18 and FB15k) from [this repo](https://github.com/thunlp/KB2E).
