# Softmax

## Define the Softmax Function
We begin by defining the **softmax** function, which is often used in machine learning, especially in classification problems, to convert raw scores (logits) into probabilities:

The softmax function is defined as:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

Where:
- $z_i$ is the $i$-th element of the input vector $z = [z_1, z_2, \ldots, z_n]$.
- $e^{z_i}$ is the exponential function applied to $z_i$, which ensures that the output probabilities are positive.
- $\sum_{j=1}^{n} e^{z_j}$ is the normalization term that ensures that the sum of the output probabilities equals 1, making it suitable for interpreting the outputs as probabilities.

### Why Use Softmax?
The softmax function maps a vector of arbitrary real numbers into a vector of probabilities:

- It is especially useful for multi-class classification problems where each element of the output vector represents the probability of a specific class.
- The sum of all probabilities for the output vector is equal to 1:

$$\sum_{i=1}^{n} \text{softmax}(z_i) = 1$$

This property makes it ideal for predicting class membership.

## Plot the Values
To better understand the behavior of the softmax function, we can create a plot that visualizes how the output probabilities change as we modify the input values $z_i$:

1. **Input Range:** Define a range of input values $z$ (e.g., $z = [-3, -2, -1, 0, 1, 2, 3]$).
2. **Apply Softmax:** Apply the softmax function to this range:

   $$\text{softmax}(z) = \left[ \frac{e^{z_1}}{\sum_{j=1}^{n} e^{z_j}}, \frac{e^{z_2}}{\sum_{j=1}^{n} e^{z_j}}, \ldots, \frac{e^{z_n}}{\sum_{j=1}^{n} e^{z_j}} \right]$$

3. **Visualize:** Create a plot of these values to see how the probabilities change. This plot will help illustrate how softmax emphasizes larger input values while diminishing the influence of smaller ones, creating a "winner-takes-all" effect.

## Create Blobs
We will be creating **blobs** for our data samples, each blob centered around specific coordinates:

- Blobs represent clusters of data points that are generated using a distribution (e.g., Gaussian distribution).
- For each blob, we specify a center and standard deviation:

   $$\text{Blob Center: } \mu = (\mu_x, \mu_y)e$$

   $$\text{Standard Deviation: } \sigma$$

- Using these parameters, data points are sampled from a normal distribution:

   $$x \sim \mathcal{N}(\mu_x, \sigma), \quad y \sim \mathcal{N}(\mu_y, \sigma)$$

This step is crucial for visualizing how different models classify data in a multi-class setting.

## Define Unpreferred Model
We will define a model with **3 layers**:

- **Layer 1:** 25 neurons with ReLU activation.
- **Layer 2:** 15 neurons with ReLU activation.
- **Layer 3:** 4 neurons with **softmax** activation.

The softmax function in the final layer outputs class probabilities:

   $$\hat{y} = \text{softmax}(Wx + b)$$

Where:
- $W$ is the weight matrix of the final layer.
- $x$ is the input from the previous layer.
- $b$ is the bias vector.
- $\hat{y}$ represents the output probabilities for each class.

## Define Preferred Model
This preferred model will use a **linear activation** function for the last layer. This allows the `from_logits=True` option in the loss function for better numerical stability:

- **Layer 1:** 25 neurons with ReLU activation.
- **Layer 2:** 15 neurons with ReLU activation.
- **Layer 3:** 4 neurons with **linear activation** (no activation).

Using logits (i.e., raw scores) directly helps avoid numerical instability when calculating cross-entropy loss. The loss function internally applies the softmax function for better precision:

   $$\text{Cross-Entropy Loss: } -\sum_{i=1}^{n} y_i \log(\text{softmax}(z_i))$$

   Where:
   - $y_i$ is the true label for class $i$.
   - $z_i$ is the raw score (logit) for class $i$ from the model's output.

## Conclusion
By comparing the outputs from the preferred and unpreferred models:

- The unpreferred model directly uses softmax in the output layer, which may lead to **numerical instability** when dealing with very large or small values.
- The preferred model, using `from_logits=True`, provides **better stability** and **precision**, particularly during training.

You can see that the outputs of the preferred model yield more distinct probabilities, making it easier for the model to distinguish between different classes.
