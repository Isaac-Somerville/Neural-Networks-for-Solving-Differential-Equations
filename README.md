# Neural Networks for Solving Differential Equations

## An exploration into artificial neural networks and how they can be applied to the study of differential equations.

This project was undertaken as a dissertation for a Master's programme in Mathematics at Durham University, where it attained a final grade of 85%. The report will be comprehensible for anyone with a solid grounding in undergraduate maths and Python programming; no prior knowledge of neural networks, PyTorch or numerical solutions to differential equations are required. An overview of the report structure is given below.

### Abstract

Finding numerical solutions to differential equations is crucial to many scientific disciplines. In
the 1990s, a new method was proposed which utilises neural networks to approximate the solution
function of a differential equation. In this dissertation, we explore the efficacy of this method
in a variety of different examples. We also examine the effect of varying different aspects of our
neural network’s structure and training methodology. Finally, we consider a recent extension of
the original method, and another technique by which neural networks can be used to estimate
unknown parameters in a differential equation.

### Chapter 1: Introduction

We give a brief summary of the histories of both ANNs and differential equations, and a hint at how the two were combined by the Lagaris method in the 1990s. We then outline the structure of the rest of the report.

### Chapter 2: Neural Networks

We begin by understanding some basic notions in the study of neural networks, with our desired
applications in mind. We see the fundamental components that make up a neural network,
and a common method of training them. Then, we explore automatic differentiation, the
computational technique behind a crucial part of their training. 

### Chapter 3: Function Approximation

We explain why neural networks are suited to the task of function approximation, and then give
an introduction to PyTorch, the machine learning library used throughout this project. We then
demonstrate a basic example of using a neural network in PyTorch to approximate sin(x).

### 

Moshe Leshno et al. “Multilayer feedforward networks with a nonpolynomial activation function can approximate any function”. In: Neural Networks 6.6 (1993), pp. 861–867. issn: 0893-6080. doi: https://doi.org/10.1016/S0893-6080(05)80131-5. url: https://www.sciencedirect.com/science/article/pii/S0893608005801315.

I.E. Lagaris, A. Likas, and D.I. Fotiadis. “Artificial neural networks for solving ordinary and
partial differential equations”. In: IEEE Transactions on Neural Networks 9.5 (1998), pp. 987–
1000. doi: 10.1109/72.712178. url: https://arxiv.org/abs/physics/9705023.

Cedric Flamant, Pavlos Protopapas, and David Sondak. “Solving Differential Equations Using
Neural Network Solution Bundles”. In: CoRR abs/2006.14372 (2020). arXiv: 2006 . 14372.
url: https://arxiv.org/abs/2006.14372.

Maziar Raissi, Paris Perdikaris, and George E. Karniadakis. “Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations”. In: CoRR
abs/1711.10566 (2017). arXiv: 1711.10566. url: http://arxiv.org/abs/1711.10566.
