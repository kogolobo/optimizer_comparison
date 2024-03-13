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

$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} * \nabla_\theta J(\theta, x,y)$.

Here, $v_t$ is the sum of squared gradients. $\frac{\partial J}{\partial \theta}$ is the gradient of the cost function with respect to the parameters, $\eta$ is the learning rate, and $\beta$ is the moving average parameter, set to $0.9$ by default.

**Adam** combines RMSProp and momentum. It uses both the sum of gradients and the sum of squared gradients. It also applies a bias-correction on the first and second moments: $m_t, v_t$.

$m_t = \beta_1 * m_{t-1} + (1-\beta_1)*\nabla_\theta J(\theta, x,y)$

$v_t = \beta_2 * v_{t-1} + (1-\beta_2) * (\nabla_\theta J(\theta, x,y))^2$

$\hat{m_t} = \frac{m_t}{1 - \beta_1} \quad \hat{v_t} = \frac{v_t}{1- \beta_2}$

$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v_t} + \epsilon}} * \hat{m_t}$

## Experiments
We run our experiments on Nvidia A100 GPU in the Google Collab environment. We use a batch size of 512 and a constant learning rate, which we tune for each task. We use the accuracy metric to evaluate our models.

### CIFAR-10
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datset contains $60000$ 32x32 pixel RGB images in 10 classes with $6000$ images per class. The dataset is composed of $50000$ training and $10000$ test images. The data classes are mutually exclusive.

We finetune a ResNet-50 checkpoint with learning rate $10^{-3}$. From the training loss curves in Figure 1, we see that RMSProp has the worst training stability while AdaGrad is the most stable. We further observe that SGD has the slowest convergence while Adam converges most rapidly.

We achieve the best classification accuracy of $85.04\%$ with Adam. Based on the evaluation curves in Figure 2 and Figure 3, we see that Adagrad and Adam consistently outperform the other optimizers with respect to both evaluation set loss and prediction accuracy. We again notice that Adagrad is most stable, while Adam is most performant.

Based on the loss and accuracy curves, we observe that the Adam optimizer achieves the highest accuracy while having the fastest convergence in both training and test loss.

![cifar_10_train_loss](https://github.com/kogolobo/optimizer_comparison/assets/44957968/0820d34b-2a9d-4c7a-99e5-cdf27967bd5b)
Figure 1: CIFAR-10 training loss curves. RMSProp is the most unstable, while Adagrad is the most stable optimizer. Adam has the fastest convergence.

![cifar_10_eval_loss](https://github.com/kogolobo/optimizer_comparison/assets/44957968/74203cd0-bc08-4c89-b3cf-36280bad6e86)
![cofar_10_accuracy](https://github.com/kogolobo/optimizer_comparison/assets/44957968/dedcb9dd-fcfd-47c6-a0d1-c1f9a93f4e9e)
Figures 2 and 3: CIFAR-10 evaluation loss and accuracy curves. Adagrad is most stable, while Adam is most performant.


### SST2
The Stanford Sentiment Treebank dataset consists of 67,349 training and 872 test phrases extracted from movie reviews, along with human-judged binary sentiment annotations (positive/negative). The task is to predict the sentiment of each phrase.

We finetune DistilBERT on the task with a learning rate of $10^{-2}$ for SGD and $10^{-4}$ for other optimizers. From the training loss curves in Figure 4, we see that RMSProp and SGD are the least stable, while Adam is both the most stable and has the fastest convergence.

Based on evaluation results in Figures 5 and 6, we see that SGD has the least training loss while Adam and Adagrad have the best accuracy. We also note that Adam reaches peak accuracy of $90.83\%$ early in training. This is also when its evaluation loss starts to increase. A similar trend is present for Adagrad and RMSProp as well. We believe this is because Adam, Adagrad, and RMSProp fit the model to the data early in training, and begin to overfit after that. 
 
We also highlight that we have selected learning rates to be $10^{-2}$ for SGD and $10^{-4}$ for other optimizers because otherwise the optimizers would not fit the model to the data. Taking this together with potential overfitting of non-SGD optimizers, we decide that SGD makes much smaller updates than the other optimizers in this setting.

![sst2_train_loss](https://github.com/kogolobo/optimizer_comparison/assets/44957968/50896a5d-db1a-425c-8d10-ca8abf7b94a3)
Figure 4: SST2 training loss curves. RMSProp and SGD are the most unstable. Adam has the fastest convergence.
 
![sst2_eval_loss](https://github.com/kogolobo/optimizer_comparison/assets/44957968/a57d13eb-9d2c-4fae-ba55-01e874ecd161)

![sst2_acc](https://github.com/kogolobo/optimizer_comparison/assets/44957968/91b8683e-6909-42d1-a58f-fa5b6f77bbe8)
Figures 5 and 6: SST2 evaluation loss and accuracy curves. SGD has the lowest evaluation loss, but Adam and Adagrad have the highest accuracy. The increasing evaluation loss indicates potential overfitting.

### MNLI
The Multi-Genre Natural Language Inference Corpus is a dataset of sentence pairs with a crowd-sourced textual annotation labels. The first sentence in the pair is the premise and the second one is the hypothesis. The task is to decide whether the premise entails the hypothesis: *contradiction* (0), *neutral* (1), and *entailment* (2). The test set is split into the "matched" category which resembles the distribution of the training data and the "mismatched" category which does not. We evaluate only on the "matched" test category. The training set has 392,702 examples, and the "matched" test set has 9,815 examples.

Before comparing the optimizers, we want to address why RMSProp is not included in any of the figures below. RMSProp proved to be too unstable and was not able to decrease the training loss. We tried multiple learning rates to no avail.

We finetune DistilBERT on the task with learning rate of $10^{-2}$ for SGD and $10^{-5}$ for the other optimizers. From the training loss in Figure 7 we see that all optimizers are similarly unstable. However, Adam seems to decrease fastest.

Looking at the plots in Figures 8 and 9, we note that Adam has the least loss and the highest test accuracy of $80.21\%$. Adagrad starts off better than SGD for both evaluation loss and training accuracy. Over time SGD catches up and beats it on both metrics.

![mnli_train_loss](https://github.com/kogolobo/optimizer_comparison/assets/44957968/677681aa-4533-419c-ae16-8eafe410eb0c)
Figure 7: MNLI training loss. All the optimizers are oscillating while decreasing. Adam is doing slightly better than the others.

![mnli_eval_loss](https://github.com/kogolobo/optimizer_comparison/assets/44957968/78c6a54a-0203-4018-bcc3-13d2bf52f78e)

![mnli_acc](https://github.com/kogolobo/optimizer_comparison/assets/44957968/45cbb379-b8cd-4112-a37a-fade36faab1c)
Figures 8 and 9: MNLI evaluation loss and accuracy curves. Adam has the lowest evaluation loss and the highest training accuracy.

### Conclusion
In the course of the comparison, we observe that SGD and RMSProp performs worse than the other two optimization methods. Specifically, SGD makes the least magnitude updates per step, while RMSProp updates too much, which leads to instability. SGD trains consistently across all the examples, but does not achieve the accuracy of AdaGrad or Adam. Adagrad and ADAM perform similarly on CIFAR-10 and SST2, however Adam is notably better on MNLI which is by far the most challenging dataset.
