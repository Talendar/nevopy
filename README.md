<img src="https://github.com/Talendar/nevopy/blob/master/docs/imgs/nevopy.png?raw=true" width="180" alt="NEvoPy logo">

<h2> Neuroevolution for Python </h2>

![Python versions](https://img.shields.io/pypi/pyversions/nevopy)
[![License](https://img.shields.io/github/license/Talendar/nevopy)](https://github.com/Talendar/nevopy/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/nevopy)](https://pypi.org/project/nevopy/)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://nevopy.readthedocs.io/en/latest/index.html)

*NEvoPy* is an open source neuroevolution framework for Python. It provides a
simple and intuitive API for researchers and enthusiasts in general to quickly
tackle machine learning problems using neuroevolutionary algorithms. *NEvoPy* is
optimized for distributed computing and has compatibility with TensorFlow.

Currently, the neuroevolutionary algorithms implemented by *NEvoPy* are:

  * **NEAT** (NeuroEvolution of Augmenting Topologies), a powerful method by
    Kenneth O. Stanley for evolving neural networks through complexification;
  * the standard fixed-topology approach to neuroevolution, with support to
    TensorFlow and deep neural networks.

Note, though, that there's much more to come!

In addition to providing high-performance implementations of powerful
neuroevolutionary algorithms, such as NEAT, *NEvoPy* also provides tools to help
you more easily implement your own algorithms.

Neuroevolution, a form of artificial intelligence that uses evolutionary
algorithms to generate artificial neural networks (ANNs), is one of the most
interesting and unexplored fields of machine learning. It is a vast and
expanding area of research that holds many promises for the future.

<h2> Installing </h2>

To install the current release, use the following command:

```
$ pip install nevopy
```

<h2> Getting started </h2>

To get started with *NEvoPy*,
[this](https://nevopy.readthedocs.io/en/latest/nevopy_overview.html) quick
guide is especially useful. You should also check out some examples available
[here](https://github.com/Talendar/nevopy/tree/master/examples).
The [XOR example](https://colab.research.google.com/github/Talendar/nevopy/blob/master/examples/xor/nevopy_xor_example.ipynb)
is particularly instructive.

The project's documentation is available on *Read the Docs*. You can access it
through this [link](https://nevopy.readthedocs.io/en/latest/index.html).

<p>
  <a href="https://github.com/Talendar/nevopy/blob/master/examples/flappy_bird/flappy_bird_simple_neat.py">
    <img align="center" 
       src="https://github.com/Talendar/nevopy/blob/r0.1.0/docs/imgs/flappy_bird.gif?raw=true" 
       height="250"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/Talendar/nevopy/blob/master/examples/lunar_lander/lunar_lander_neat.py">
    <img align="center" 
       src="https://github.com/Talendar/nevopy/blob/r0.1.0/docs/imgs/lunar_lander.gif?raw=true" 
       height="250"/>
  </a> 
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/Talendar/nevopy/blob/master/examples/cart_pole/cart_pole_neat.py">
    <img align="center" 
       src="https://github.com/Talendar/nevopy/blob/r0.1.0/docs/imgs/cart_pole.gif?raw=true" 
       height="250"/>
  </a> 
</p>

<h2> Citing </h2>

If you use *NEvoPy* in your research and would like to cite the *NEvoPy*
framework, here is a Bibtex entry you can use. It currently contains only the
name of the original author, but more names might be added as more people
contribute to the project. Also, feel free to contact me (Talendar/Gabriel) to
show me your work - I'd love to see it.

```
@misc{nevopy,
  title={ {NEvoPy}: A Neuroevolution Framework for Python},
  author={Gabriel Guedes Nogueira},
  howpublished={\url{https://github.com/Talendar/nevopy}},   
}
```
