# 📘 Machine Learning Algorithms From Scratch

This repository contains from-scratch implementations of common supervised learning algorithms using only NumPy. The goal is to understand the inner workings of these models.

---

## 📈 Linear Regression

### Gradient Descent Update Rules:
- Prediction: $\hat{y} = X \cdot w + b$
- Gradient w.r.t. weights: $\frac{\partial J}{\partial w} = \frac{1}{m} \cdot X^T \cdot (\hat{y} - y)$
- Gradient w.r.t. bias: $\frac{\partial J}{\partial b} = \frac{1}{m} \cdot \sum(\hat{y} - y)$
- Parameter update:  
  $w := w - \alpha \cdot \frac{\partial J}{\partial w}$  
  $b := b - \alpha \cdot \frac{\partial J}{\partial b}$

### Evaluation Metrics:
- Mean Squared Error (MSE):  
  $\text{MSE} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)^2$
- R² Score:  
  $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

---

## 🤖 Logistic Regression

### Gradient Descent (Same as Linear Regression but with Sigmoid):
- Linear combination: $z = X \cdot w + b$
- Sigmoid activation: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Prediction: $\hat{y} = \sigma(z)$
- Loss gradient same as linear regression with $\hat{y} = \sigma(z)$

### Evaluation Metric:
- Accuracy:  
  $\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}$

---

## 🧠 Naive Bayes Classifier (Gaussian)

### Gaussian Probability Density Function:
- $P(x_i | y) = \frac{1}{\sqrt{2 \pi \cdot \sigma^2}} \cdot \exp\left(-\frac{(x_i - \mu)^2}{2 \sigma^2}\right)$

### Posterior Probability:
- $\log P(y | x) \propto \log P(y) + \sum_i \log P(x_i | y)$

### Evaluation Metric:
- Accuracy:  
  $\text{Accuracy} = \frac{\text{Correct predictions}}{\text{Total predictions}}$

---

## 🌳 Decision Tree Classifier

### Entropy:
- $H(y) = -\sum p_i \log_2(p_i)$

### Information Gain:
- $IG = H(y) - \left( \frac{n_L}{n} H(y_L) + \frac{n_R}{n} H(y_R) \right)$

### Evaluation Metric:
- Accuracy:  
  $\text{Accuracy} = \frac{\text{Correct predictions}}{\text{Total predictions}}$

---

## 🌲 Decision Tree Regressor

### Mean Squared Error (MSE):
- $\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$

### Information Gain:
- $IG = \text{MSE}_{\text{parent}} - \left( \frac{n_L}{n} \text{MSE}_L + \frac{n_R}{n} \text{MSE}_R \right)$

---

## 📊 K-Means Clustering

### 📐 Euclidean Distance Formula

Used to measure the distance between a data point and a centroid during clustering:

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

