# formula

## Basic

$f(\theta) = \mathop{{}\mathbb{E}}X$

Another form for above function. 

$\displaystyle f(\theta) = \sum_{x \in X}p(x|\theta)x$

Where
- $\theta$ is the parmeter
- X is Random variable and it's distribution is function of $\theta$, we may write it's probability distribution function as $p(x|\theta)$
- $\mathop{{}\mathbb{E}}$ is Expectation
- The expection should be a normal function $f(\theta)$

How to understand the formula:
-  Different parameter $\theta$ give different distribution for random variable $X$
-  For examples: a simple X distribuiton, $x \in X$, and $x$ has three options, $\{x_1 = 0, x_2=1, x_3 = 2\}$. Probability distribution is function of $\theta$, $p(x|\theta)$ 
   -  $\displaystyle p(x|\theta)=\frac{x+\theta}{3\theta+3}$ where $\theta \ge 0$

Note: My understanding about $p(x|\theta)$ is actually the function with $x$ and $\theta$ as variables.  The "$|$" instead of "$,$" can express extra meaning that $x$ is also a sample from $X$ random variable. 


## Derivitive toward $\theta$ with random variable

(1) $\displaystyle\frac{df(\theta)}{d\theta}=\frac{d\mathop{{}\mathbb{E}}X}{d\theta}$
$\displaystyle =\frac{d\sum_{x \in X}p(x|\theta)x}{d\theta}$

where X follows distribution given by $p(x|\theta)$

(2) $\displaystyle =\frac{d\sum_{x \in X}p(x|\theta)x}{d\theta}$

(3) $\displaystyle =\sum_{x \in X}\frac{dp(x|\theta)}{d\theta}x$

(4) $\displaystyle =\sum_{x \in X}p(x|\theta)\frac{d\ln p(x|\theta)}{d\theta}x$

(5) $\displaystyle  =\mathop{{}\mathbb{E}}\frac{d\ln p(X|\theta)}{d\theta}X$

$\frac{d}{d\theta}$ can be written as $\nabla_\theta$, so

(6) $\displaystyle \nabla_\theta f(\theta)=\mathop{{}\mathbb{E}}\nabla_\theta\ln p(X|\theta)X$ 

$X\sim p$ Radom variable $X$ follow $p(x|\theta)$ distribution. 


### Case analyse

In the case of: $\displaystyle p(x|\theta)=\frac{x+\theta}{3\theta+3}$ 

where $\theta \ge 0$ and $x \in X$, $x$ has three options, $\{x_1 = 0, x_2=1, x_3 = 2\}$.

$\displaystyle f(\theta) = \mathop{{}\mathbb{E}}X=\sum_{x \in X}p(x|\theta)x$

$\displaystyle=\frac{(0+\theta)\cdot 0+(1+\theta)\cdot 1+(2+\theta)\cdot 2}{3\theta+3}$

$\displaystyle=\frac{5+3\theta}{3\theta+3}$



<!-- $f(\theta)=\mathop{\mathop{{}\mathbb{E}}}X$
$=\mathop{\mathop{{}\mathbb{E}}}p(X|\theta).\frac{X}{p(X|\theta)}$ -->


# Stochastic Gradient
```python
import torch

theta = torch.tensor(0., requires_grad=True)
X = torch.tensor([0,1,2], requires_grad=False)
with torch.no_grad():
    probs_dist = torch.distributions.Categorical((X+theta)/(3*theta+3))
    X_samples = probs_dist.sample([10000])
    # probs_dist.log_prob(torch.tensor([0,1,2]))

X_samples = probs_dist.sample([1000000]) # use monte carlo to estimate expectation
def p(x, theta):
    return (x+theta)/(3*theta+3) #p(x|theta)
# print(X_samples)
dd = torch.log(p(X_samples, theta))*X_samples #ln(p(x|theta))*x
(dd.mean()).backward() # backward to get the gradient of ln(p(x|theta))*x, mean is to estimate expectation
print(theta.grad) # the gradient, should be -2/3
```