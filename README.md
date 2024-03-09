# Optimizer Comparison

We compare four popular optimization algorithms - Stochastic Gradient Descent (SGD), Adaptive Gradient Method (Adagrad), Root Mean Squared Propagation (RMSProp), and Adaptive Moment Estimation (ADAM) - across three large publicly available datasets: [CIFAR-10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html) for image classification, and [SST-2](https://pytorch.org/text/stable/datasets.html#sst2) and [MNLI](https://pytorch.org/text/stable/datasets.html#mnli) for natural language processing tasks. 

We observe that RMSProp exhibits unstable convergence and consistently underperforms compared to the other methods.  Adagrad and Adam perform similarly on CIFAR-10 and SST-2, but Adam notably outperforms Adagrad on the more challenging MNLI dataset.

## Getting Started
To reproduce our experiments, please install the dependencies in `requirements.txt` and run the command:

`main.py --model [distilbert|resnet] --dataset [cifar10|sst2|mnli] --lr <sety value as described below>`

The command will run all optimizers on a given task/model combination. Note, that `resnet` only supports `cifar10` data, while `distilbert` only supports `sst2|mnli` data.

## Background

### Optimization in Deep Learning
The task of fitting a deep learning model to data $(x^{(i)}, y^{(i)})_{i=1}^N \overset{\text{i.i.d.}}{\sim} \mathcal{P}$ comes down to selecting an optimal set of model paramters $\theta$ with respect to training loss $J(\theta, x,y)$, i.e., $\theta^* = \arg\min \mathbb{E} \left[J(\theta, x,y)\right]  $. 

At each training iteration $t$, the parameters $\theta$ of a neural network are updated: $\theta_t \to \theta_{t+1}$. Typically, this uses the gradient of the training loss with respect to the parameters: $\nabla_\theta J(\theta, x, y)$, together with additional components. Further, we cannot feasibly use the full dataset $(x^{(i)}, y^{(i)})_{i=1}^N$ at each training iteration. Rather, at each iteration, we randomly sample a batch of $n$ examples $(x,y) = (x^{(i:i+n)}, y^{(i:i+n)})$ and compute weight updates with respect to only that data.

### Optmization Methods
In selecting optimization algorithms to benchmark, we draw inspiration from [Sebastian Ruder's Blog](https://www.ruder.io/optimizing-gradient-descent/). We compare the following methods:

**SGD** proposes a simple update rule, which uses only one hyperparameter $\eta$ -- learning rate: 

$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t, x,y)$.

**Adagrad** keeps track of the sum of gradient squared and uses that to adapt the gradient in different directions. The idea is that the learning rate then adapts for each weight in the model. It also decays the learning rate $\eta$ for parameters in proportion to their update history:

$v_{t} = v_{t-1} + (\nabla_\theta J(\theta, x,y))^2$

$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} * \nabla_\theta J(\theta, x,y)$

Looking at the update history we see that if the accumulated gradient is small then $v$ will be small. This leads to a big learning rate $\eta$

**RMSProp** handles the learning rate by maintaining a moving average of the squares of gradients for each weight and dividing the learning rate by that. This is similar to Adagrad. The only difference is that it adds a decay factor to the gradient squares. This is supposed to make RMSProp much quicker than Adagrad since Adagrad decays the learning rate very aggressively.

$v_t = \beta * v_{t-1} + (1-\beta)(\nabla_\theta J(\theta, x,y))^2$

$\theta_{t+1} = \theta_t - \frac{n}{\sqrt{v_t + \epsilon}} * \nabla_\theta J(\theta, x,y)$.

Here, $v_t$ is the sum of squared gradients. $\frac{\partial J}{\partial \theta}$ is the gradient of the cost function with respect to the parameters, $\eta$ is the learning rate, and $\beta$ is the moving average parameter, set to $0.9$ by default.

**Adam** combines RMSProp and momentum. It uses both the sum of gradients and the sum of squared gradients. It also applies a bias-correction on the first and second moments: $m_t, v_t$.

$m_t = \beta_1 * m_{t-1} + (1-\beta_1)*\nabla_\theta J(\theta, x,y)$

$v_t = \beta_2 * v_{t-1} + (1-\beta_2) * (\nabla_\theta J(\theta, x,y))^2$

$\hat{m_t} = \frac{m_t}{1 - \beta^t_1} \quad \hat{v_t} = \frac{v_t}{1- \beta^t_2}$

$\theta_{t+1} = \theta_t - \frac{n}{\sqrt{\hat{v_t} + \epsilon}} * \hat{m_t}$
