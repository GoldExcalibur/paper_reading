# adversarial trainning
## Explaining and Harnessing Adversarial Examples
[论文链接](https://arxiv.org/abs/1412.6572)，年份2014  


## Motivation
Trained neural networks will misclassify **adversarial examples** with high confidence. 
- adversarial examples: inputs formed by applying small but intentionally worst-case pertubations to examples from the dataset.

Models only behave well on **natural data**, but badly in space that do not have high probability in the data distribution.  
- Flawed: Regard **Euclidean distance** in **convolutional feature space** as perceptual distance.

Goal: **resist adversarial examples** but also try to **maintain state-of-the-art accuracy** on clean input data.

## Speculative Explanations:   
- Previous work: overfitting, model non-linearity, insufficient model averaging, insufficient model regularity. (unnecessary)

- This paper: linear behaviour of high dimension spaces (sufficient)

### Linear Explanation of adversarial examples
- limited precision of an individual input feature. For example, an 8-bit image will discard information below $\frac{1}{255}$   
For small $\epsilon$ below the precision of input feature, we set pertubation $\eta$ = $\epsilon sgn(w)$. Then we have:

$$ 
w^t \tilde{x} = w^tx + w^t\eta, s.t. ||\eta||_{\infty} < \epsilon
$$ 
If the average magnitude of $w$ is $m$, then the total increase is $\epsilon mn$. ($n$ is the dimension of input feature). The change in activation grows linearly with $n$.   
A linear model is forced to exclusively attend to **the signal that aligns most closely with its weights**, even if multiple signals are present and other signals have much greater amplitude.

### Linear Pertubation of NonLinear Models
- **LSTMs, ReLUs and maxout networks** are designed to behave in linear ways, for the ease of optimization.
- More non-linear models like sigmoid networks are carefully tuned to spend most of the time in the non-saturating, more linear regime for the same reason.
#### fast gradient method
notations
- $\theta$ model parameters
- $x$ the input to the model
- $y$ the targets associated with $x$ 
- $J(\theta, x, y)$ the cost used to train the neural network.   

Linearize the cost function $J$ around the current value of $\theta$, obtaining an max norm pertubation of 
$$ 
\eta = \epsilon sign(\nabla_{x}J(\theta, x, y))
$$
$\epsilon$ should be small enough. Since $\eta$ is aligned to the sign of the gradient, the loss function will increase so that the classification result of trained model might change. 

#### other simple methods
Rotating $x$ by a **small angle in the direction of gradient** reliably produces adversarial examples.

### Adversarial Training of Linear Models versus Weight Decay
#### Logistic Regression
Target $y \in \{1, -1\}$, sigmoid function $\sigma(z) = \frac{1}{1 + \exp(-z)} = 1 - \sigma(-z)$.  

 Then we have $P(y = 1|x) = \sigma(w^Tx+b) = \sigma(y(w^Tx+b))$. The generalization cost can be written as follows:  
$$
\mathbb{E}_{x, y\sim p_{data}}\zeta(-y(w^Tx+b))
$$
where $\zeta(z) = \log(1 + \exp(z))$. The above cost can be derived from the negative log likelihood of sample $S$. 
$$
-\log L(w,b|\{x_i, y_i\}) = -\sum_i \log P(y_i|x_i) 
$$
$$
 = -\sum_i \log(\sigma(y_i(w^Tx_i + b))) 
$$
$$
= \sum_i \log(1 + \exp(y_i(w^Tx_i + b)))
$$
Then we direcly apply the fast gradient method to train on adversarial examples. The sign of the gradient is just $-ysign(w)$. The adversarial version of logistic regression is therefore to minimize   
$$
\mathbb{E}_{x,y\sim p_{data}}\zeta(\epsilon ||w||_1 + y(- w^Tx - b))
$$
Compared to L1 weight decay, 

## Appendix


