## adversarial trainning
### Explaining and Harnessing Adversarial Examples
[论文链接](https://arxiv.org/abs/1412.6572)，年份2014  


#### Motivation
Trained neural networks will misclassify **adversarial examples** with high confidence. 
- adversarial examples: inputs formed by applying small but intentionally worst-case pertubations to examples from the dataset.

Models only behave well on **natrual data**, but badly in space that do not have high probability in the data distribution.  
- Flawed: Regard **Euclidean distance** in **convolutional feature space** as perceptual distance.


Speculative Explanations:   
- Previous work: overfitting, model non-linearity, insufficient model averaging, insufficient model regularity. (unnecessary)

- This paper: linear behaviour of high dimension spaces (sufficient)

