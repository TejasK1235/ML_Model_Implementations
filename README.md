# 📘 Machine Learning Algorithms From Scratch

This repository contains from-scratch implementations of common supervised learning algorithms using only NumPy. The goal is to understand the inner workings of these models.

---

## 📈 Linear Regression

### Gradient Descent Update Rules:
- For each epoch:
  $$
  \hat{y} = X \cdot w + b
  $$
  $$
  \frac{\partial J}{\partial w} = \frac{1}{m} \cdot X^T \cdot (\hat{y} - y)
  $$
  $$
  \frac{\partial J}{\partial b} = \frac{1}{m} \cdot \sum (\hat{y} - y)
  $$
  $$
  w := w - \alpha \cdot \frac{\partial J}{\partial w}
  \quad
  b := b - \alpha \cdot \frac{\partial J}{\partial b}
  $$

### Evaluation Metrics:
- Mean Squared Error (MSE):
  $$
  \text{MSE} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)^2
  $$
- R² Score:
  $$
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  $$

---

## 📉 Logistic Regression

### Sigmoid Activation:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### Gradient Descent Update Rules:
- For each epoch:
  $$
  \hat{y} = \sigma(X \cdot w + b)
  $$
  $$
  \frac{\partial J}{\partial w} = \frac{1}{m} \cdot X^T \cdot (\hat{y} - y)
  $$
  $$
  \frac{\partial J}{\partial b} = \frac{1}{m} \cdot \sum (\hat{y} - y)
  $$

### Evaluation Metric:
- Accuracy:
  $$
  \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}
  $$

---

## 🧠 Naive Bayes Classifier

### Gaussian Probability Density Function:
$$
P(x_i | y) = \frac{1}{\sqrt{2\pi \cdot \sigma^2}} \cdot \exp\left( - \frac{(x_i - \mu)^2}{2 \cdot \sigma^2} \right)
$$

### Log Posterior Probability:
$$
\log P(y|x) \propto \log P(y) + \sum_{i=1}^{n} \log P(x_i | y)
$$

### Evaluation Metric:
- Accuracy:
  $$
  \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}
  $$

---

## 🌳 Decision Tree Classifier

### Entropy:
$$
H(y) = - \sum_{i=1}^{k} p_i \log_2(p_i)
$$

### Information Gain:
$$
IG = H(\text{parent}) - \left( \frac{n_{left}}{n} H(\text{left}) + \frac{n_{right}}{n} H(\text{right}) \right)
$$

### Evaluation Metric:
- Accuracy:
  $$
  \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}
  $$

---

## 🌲 Decision Tree Regressor

### Mean Squared Error (MSE):
$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2
$$

### Information Gain (based on MSE):
$$
IG = \text{MSE(parent)} - \left( \frac{n_{left}}{n} \cdot \text{MSE(left)} + \frac{n_{right}}{n} \cdot \text{MSE(right)} \right)
$$

---

## ✅ Summary

Each algorithm is implemented from scratch using NumPy only. For more details, check the corresponding `.py` files in this repository.

