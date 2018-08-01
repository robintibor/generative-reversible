Training Generative Reversible Networks
=======================================
This repository contains code accompanying the paper `Training Generative Reversible Networks <https://arxiv.org/abs/1806.01610>`_.

Installation
------------

1. Install `Pytorch 0.4  <http://pytorch.org/>`_.
2. Install other dependencies such as numpy, matplotlib, seaborn, etc.
3. Add this repository to your PYTHONPATH

Run
------

To reproduce the CelebA results, first run the notebooks under notebooks/celeba:

1. `Only_Clamp.ipynb <https://github.com/robintibor/generative-reversible/blob/master/notebooks/celeba/Only_Clamp.ipynb>`_
2. `Continue_Adversarial.ipynb <https://github.com/robintibor/generative-reversible/blob/master/notebooks/celeba/Continue_Adversarial.ipynb>`_

Running each for 250 epochs should definitely be enough to get similar results as the ones in `notebooks/celeba/Plots.ipynb  <https://github.com/robintibor/generative-reversible/blob/master/notebooks/celeba/Plots.ipynb>`_.

To reproduce the MNIST results, run the notebook under notebooks/mnist:

1. `OptimalTransport.ipynb <https://github.com/robintibor/generative-reversible/blob/master/notebooks/mnist/OptimalTransport.ipynb>`_

You should get plots similar to the ones in `notebooks/mnist/Plots.ipynb  <https://github.com/robintibor/generative-reversible/blob/master/notebooks/mnist/Plots.ipynb>`_. Latent dimensions are arguably a bit less meaningful than in the paper, this setup could certainly be further optimized and stabilized, feel free to contact me if you are interested to discuss it.

Citing
------
If you use this code in a scientific publication, please cite us as:

.. code-block:: bibtex

  @inproceedings{schirrm_revnet_2018,
  author = {Schirrmeister, Robin Tibor and Chrabąszcz, Patryk and Hutter,
    Frank and Ball, Tonio},
  title = {Training Generative Reversible Networks},
  url = {https://arxiv.org/abs/1806.01610},
  booktitle = {ICML 2018 workshop on Theoretical Foundations and Applications of Deep Generative Models},
  month = {jul},
  year = {2018},
  keywords = {Generative Models, Reversible Networks, Autoencoders},
  }
