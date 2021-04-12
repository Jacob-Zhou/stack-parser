# Stack Parser (Share-Loose)

The implementation of "Is POS Tagging Necessary or Even Helpful for Neural Dependency Parsing?".

## Requirements

```txt
python == 3.7.0
pytorch == 1.0.0
```

## Datasets

TODO

## Performance

TODO

## Usage

You can start the training, evaluation and prediction process by using subcommands registered in `parser.cmds`.

```sh
$ python run.py -h
usage: run.py [-h] {evaluate,predict,train} ...

Create the Biaffine Parser model.

optional arguments:
  -h, --help            show this help message and exit

Commands:
  {evaluate,predict,train}
    evaluate            Evaluate the specified model and dataset.
    predict             Use a trained model to make predictions.
    train               Train a model.
```

Before triggering the subparser, please make sure that the data files must be in CoNLL-X format. If some fields are missing, you can use underscores as placeholders.

Optional arguments of the subparsers are as follows:

```sh
$ python run.py train -h
usage: run.py train [-h] [--buckets BUCKETS] [--ftrain FTRAIN] [--fdev FDEV]
                    [--ftest FTEST] [--fembed FEMBED] [--device DEVICE]
                    [--seed SEED] [--threads THREADS] [--file FILE]
                    [--vocab VOCAB]

optional arguments:
  -h, --help            show this help message and exit
  --buckets BUCKETS     max num of buckets to use
  --ftrain FTRAIN       path to train file
  --fdev FDEV           path to dev file
  --ftest FTEST         path to test file
  --fembed FEMBED       path to pretrained embedding file
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --file FILE, -f FILE  path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocabulary file

$ python run.py evaluate -h
usage: run.py evaluate [-h] [--batch-size BATCH_SIZE] [--buckets BUCKETS]
                       [--include-punct] [--fdata FDATA] [--device DEVICE]
                       [--seed SEED] [--threads THREADS] [--file FILE]
                       [--vocab VOCAB]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --buckets BUCKETS     max num of buckets to use
  --include-punct       whether to include punctuation
  --fdata FDATA         path to dataset
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --file FILE, -f FILE  path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocabulary file

$ python run.py predict -h
usage: run.py predict [-h] [--batch-size BATCH_SIZE] [--fdata FDATA]
                      [--fpred FPRED] [--device DEVICE] [--seed SEED]
                      [--threads THREADS] [--file FILE] [--vocab VOCAB]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --fdata FDATA         path to dataset
  --fpred FPRED         path to predicted result
  --device DEVICE, -d DEVICE
                        ID of GPU to use
  --seed SEED, -s SEED  seed for generating random numbers
  --threads THREADS, -t THREADS
                        max num of threads
  --file FILE, -f FILE  path to model file
  --vocab VOCAB, -v VOCAB
                        path to vocabulary file
```

## Hyperparameters

| Param         | Description                             |                                 Value                                  |
| :------------ | :-------------------------------------- | :--------------------------------------------------------------------: |
| n_embed       | dimension of word embedding             |                                  100                                   |
| n_char_embed  | dimension of char embedding             |                                  50                                   |
| embed_dropout | dropout ratio of embeddings             |                                  0.33                                  |
| n_lstm_hidden | dimension of lstm hidden state          |                                  400                                   |
| n_lstm_layers | number of lstm layers                   |                                   3                                    |
| lstm_dropout  | dropout ratio of lstm                   |                                  0.33                                  |
| n_mlp_arc     | arc mlp size                            |                                  500                                   |
| n_mlp_rel     | label mlp size                          |                                  100                                   |
| mlp_dropout   | dropout ratio of mlp                    |                                  0.33                                  |
| lr            | starting learning rate of training      |                                  2e-3                                  |
| betas         | hyperparameter of momentum and L2 norm  |                               (0.9, 0.9)                               |
| epsilon       | stability constant                      |                                 1e-12                                  |
| annealing     | formula of learning rate annealing      | <img src="https://latex.codecogs.com/gif.latex?.75^{\frac{t}{5000}}"/> |
| batch_size    | number of sentences per training update |                                  200                                   |
| epochs        | max number of epochs                    |                                  1000                                  |
| patience      | patience for early stop                 |                                  100                                   |

## References

* Houquan Zhou, Yu Zhang, Zhenghua Li, Min Zhang [Is POS Tagging Necessary or Even Helpful for Neural Dependency Parsing?](https://arxiv.org/abs/1611.01734)

```txt
@inproceedings{zhou2020is,
  author    = {Houquan Zhou and
               Yu Zhang and
               Zhenghua Li and
               Min Zhang},
  editor    = {Xiaodan Zhu and
               Min Zhang and
               Yu Hong and
               Ruifang He},
  title     = {Is {POS} Tagging Necessary or Even Helpful for Neural Dependency Parsing?},
  booktitle = {Natural Language Processing and Chinese Computing - 9th {CCF} International Conference, {NLPCC} 2020, Zhengzhou, China, October 14-18, 2020, Proceedings, Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12430},
  pages     = {179--191},
  publisher = {Springer},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-60450-9\_15},
  doi       = {10.1007/978-3-030-60450-9\_15},
  timestamp = {Thu, 08 Oct 2020 12:56:06 +0200},
  biburl    = {https://dblp.org/rec/conf/nlpcc/ZhouZLZ20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```