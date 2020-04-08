# meta-learning

- Methods for meta-learning
  1. [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks]

- Dependencies
  - Python 3.6+
  - PyTorch 1.3
  - Codes are inspired by [https://towardsdatascience.com], [https://github.com/potsawee/]

### Pretrain model with multi-tasks
* Run the following command.
```
# sine regression
python regression_maml.py  # for meta-training
python regression_pretrain.py  # for normal transfer learning

# omniglot
python omniglotTrain.py -n 5 -k 1 -c 0 --no_iter 50000 --lr 0.1
```

### Reference
1. [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks]
2. [How to train your MAML]

[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks]: https://arxiv.org/abs/1703.03400
[https://towardsdatascience.com]: https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0
[How to train your MAML]: https://arxiv.org/abs/1810.09502
[https://github.com/potsawee/]: https://github.com/potsawee/maml
