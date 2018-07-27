Training Generative Reversible Networks
=======================================
This repository contains code accompanying the paper
[Training Generative Reversible Networks](https://arxiv.org/abs/1806.01610).

Run
------
To reproduce the CelebA results, first run the notebooks under notebooks/celeba:
1. [Only_Clamp.ipynb](https://github.com/robintibor/generative-reversible/blob/master/notebooks/celeba/Only_Clamp.ipynb)
2. [Continue_Adversarial.ipynb](https://github.com/robintibor/generative-reversible/blob/master/notebooks/celeba/Continue_Adversarial.ipynb)

Running each for 250 epochs should definitely be enough to get similar results as the ones in
[notebooks/celeba/Plot.ipynb](https://github.com/robintibor/generative-reversible/blob/master/notebooks/celeba/Plots.ipynb).

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


TODO
------
Optimal Transport MNIST Code