![image](docs/img/logo.png)

Magpie is a deep learning tool for multi-label text classification. It learns on the training corpus to assign labels to arbitrary text and can be used to predict those labels on unknown data. It has been developed at CERN to assign subject categories to High Energy Physics abstracts and extract keywords from them.

## Very short introduction
```
$ ls training-directory
100.txt  100.lab  101.txt  101.lab  102.txt  102.lab  ...
$ python
>>> from magpie import MagpieModel
>>> model = MagpieModel()
>>> model.init_word_vectors('/path/to/training-directory', vec_dim=100)
>>> model.train('/path/to/training-directory', ['label1', 'label2', 'label3'], nb_epochs=10)
>>> model.predict_from_text(u'Well that was quick...')
[('label1', 0.84), ('label2', 0.26), ('label3', 0.04)]
```


## Short introduction
To train the model you need to have a large corpus of labeled data in a text format. Magpie looks for `.txt` files containing the text to predict on and corresponding `.lab` files with assigned labels in separate lines. A pair of files containing the labels and the text should have the same name and differ only in extensions e.g.

```
$ ls training-directory
100.txt  100.lab  101.txt  101.lab  102.txt  102.lab  ...
```

Before you train the model, you need to build appropriate word vector representations for your corpus. In theory, you can train them on a different corpus or reuse already trained ones ([tutorial](http://rare-technologies.com/word2vec-tutorial/)), however Magpie enables you to do that as well.
```
>>> from magpie import MagpieModel
>>> model = MagpieModel()
>>> model.train_word2vec('/path/to/training-directory', vec_dim=100)
```

Then you need to fit a scaling matrix to normalize input data, it is specific to the trained word2vec representation. Here's the one liner:

```
>>> model.fit_scaler('/path/to/training-directory')
```

You would usually want to combine those two steps, by simply running:
```
>>> model.init_word_vectors('/path/to/training-directory', vec_dim=100)
```

If you plan to reuse the trained word representations, you might want to save them and pass in the constructor to `MagpieModel`next time. For the training, just type:
```
>>> model.train('/path/to/training-directory', ['label1', 'label2', 'label3'],
                test_dir='/optional/test/directory', nb_epochs=30)
```
By providing the `test_dir` argument, the model is evaluated after every epoch and displays it's current loss and accuracy. If your data doesn't fit into memory, you can also run `model.batch_train()` which has the same API, but is more memory efficient.

Trained models can be used for prediction with methods:
```
>>> model.predict_from_file('/path/to/txt/file')
[('label1', 0.97), ('label3', 0.32), ('label2', 0.02)]
>>> model.predict_from_text(u'This text should preferably be longer to give more information to the NN')
[('label3', 0.47), ('label1', 0.12), ('label2', 0.003)]
```

## Installation
The package has several dependencies, we're working to reduce this number. Before downloading Magpie, make sure to install:
 - [numpy](http://www.numpy.org/)
 - [keras](https://github.com/fchollet/keras) with Theano or TensorFlow backend.
 - [gensim](http://radimrehurek.com/gensim/)
 - [scikit-learn](http://scikit-learn.org/stable/index.html)
 - [nltk](http://www.nltk.org/install.html)

Afterwards, install the package from GitHub:
```
$ pip install git+https://github.com/jstypka/magpie.git
```

## Contact
If you have any problems, feel free to open an issue. We'll do our best to help :+1:
