# CS5782 Notes: Convolutional Neural Networks (CNNs)

## Introduction

Convolutional Neural Networks (CNNs) are a type of Neural Network that is primarily designed for processing visual signals, such as image or video-related data [1] (not exclusively on images, for example, AlphaZero). The overall structure of CNN is similar to ordinary neural networks, but it has some special properties designed specifically for image processing. Before deep diving into the architecture of CNNs, let’s first recall what Multi-Layer Perceptron (MLP) is, what MLP can do and what the limitations of MLP are in the computer vision field. 

As we have seen in the previous chapters, MLP consists of several fully connected layers, where each neuron in one layer is connected to every neuron in the previous layer. The input is passed through several nonlinear hidden layers, and the output could be numerical in a regression task or class probability in a classification task. 

However, MLPs are not well-suited for processing high-dimensional structured data like images. For example, say we want to classify a set of cat and dog images of size 128 * 128, each of which has 3 color channels as they are RGB images. The first step is to flatten the image to a vector of dimension 49152 (128*128*3). Suppose the next hidden layer has 1000 neurons (not very large compared to 49152), the number of parameters for this single layer would be 49152 * 1000 = 49152000. This is only for one layer. Imagine building a neural network with multiple layers using higher resolution images, say 1024 * 1024, the number of parameters grows quickly! Training such neural networks on GPUs is difficult or even infeasible. 

Additionally, MLP is not good at capturing spatial relations and is sensitive to translation variance. Intuitively, the position of a dog in an image should not affect the classification result, whether it’s a dog or a cat. However, since MLPs treat each input pixel as independent and fixed in position, even a small spatial shift results in significantly different input vectors. This shift can drastically affect the output, even though the images are semantically identical. 

To address these issues, CNNs were brought up! Convolutional Neural Networks have the advantage of translation invariance and locality of an image input. Translation invariance means the network's output remains the same regardless of the position of an object within the image. Locality means pixels close to each other are more correlated than distant ones. 


## Architecture Overview

The CNN architecture is designed to gradually extract increasingly abstract and complex features, starting from raw pixels to low-level features to high-level features. It consists of three main types of layers: a convolutional layer, a pooling layer, and a fully connected layer. 


1. **Convolutional Layer**: Convolutional layers perform the core operation of CNNs using filters/kernels that slide across the input to produce feature maps. A convolutional layer is essentially a set of convolutional filters and each convolutional filter creates a feature map from the input.

2. **Pooling Layer**: Pooling layers downsample the feature map and highlight the most present feature. They also reduce the feature map size and increases the receptive field.
3. **Fully Connected Layer**: Fully connected layers are exactly the same as how we used them in MLPs. Each neuron in the layer is connected to all the neurons in the previous layer. If we are using CNN in a classification task, the number of neurons in the last fully connected layer should be the same as the number of classes.

The picture below shows a complete structure and essential components of CNN. Don’t be scared! We will go through each one of them and understand how they function individually as well as together to make up CNN.


## CNN Building Blocks

After the short introduction above and motivations behind the design of CNN, let’s dive deep into the building blocks of the architecture to better understand how it excels at processing visual signals!

### Convolutional Layer

The Conv layer is the core building block of CNNs and performs the majority of the computational work. Its main function is to apply a set of filters (or kernels) to the input data to extract features like edges, shapes, and textures.

#### Components:
- **Kernel/Filter**: Filters are small spatially (in width and height), but extend through the full depth of the input volume. During the forward pass, the filter slides over the input data (convolution) and computes the dot product between the filter and the input at each position. The output is an activation map that represents the feature the filter detects (e.g., edges, colors, patterns).
  - **Example**: A first-layer filter might have a size of 5x5x3 (5 pixels wide and tall, with 3 corresponding to the 3 color channels in a RGB image).
  - This is a great resource to see how convolution happens actively!

- **Padding**: Padding ensures the output volume retains the same spatial dimensions as the input, especially when stride = 1. In the example shown below, the zero padding is simply adding rows and columns of zeros around the pixels to retain dimension. 

- **Stride**: The stride defines how much the filter moves at each step. Larger strides reduce the spatial dimensions of the output, potentially losing finer details. In Figure 2, it shows how a filter of size 3*3 slides through the input slice with a stride = 1.
  1. Stride=1: Filter moves by 1 pixel at a time, leading to larger output volumes.
  2. Stride=2: Filter moves by 2 pixels, producing smaller output volumes.


#### Dimension Calculation

> Spatial output size can be calculated based on input size, kernel size, padding, and stride.

#### Characteristics

- **Local Connectivity**: Instead of connecting neurons to every other neuron in the previous layer, local connectivity restricts each neuron to a small region of the input, i.e., the receptive field. This is a region that each neuron in the Conv layer looks at, typically equivalent to the filter size. The depth of the connectivity always matches the depth of the input volume (e.g., 3 for color images), but the spatial dimensions (width and height) are localized to the size of the filter.

- **Parameter Sharing**: Parameter sharing reduces the number of parameters by using the same set of weights for all spatial positions in a depth slice. Each depth slice in the input volume shares the same weights, which means instead of having a unique set of weights for each spatial position, one set of weights is applied across all positions in that depth slice. This leads to fewer parameters and reduces computation.


#### Types of Convolution

- 1×1, 1D, 3D convolutions
- Dilated convolutions


### Pooling Layer

Pooling operates independently on every depth slice of the input. The operation reduces the width and height of each depth slice, while the depth remains unchanged. The Pooling Layer is inserted between Convolutional Layers to:
- Reduce spatial size of the representation.
- Reduce the number of parameters and computation, which helps in faster processing and lower memory usage.
- Control overfitting by downsampling, which forces the model to focus on more prominent features rather than overfitting to small details.

In the following example, we can see how a depth slice is downsampled from size 4*4 to a size of 2*2 by max pooling. Notice how only spatial dimension is reduced (256*256 -> 128*128) while the depth is maintained (64 -> 64). 


### Normalization Layer (often BatchNorm)

Helps stabilize training by normalizing activations across mini-batches.

### ReLU Layer (Activation Function)

ReLU is a very widely used activation function to introduce non-linearities between layers. It is defined as 

$$\text{ReLU}(x) = \max(0, x)$$

As ReLU is essentially an element-wise max() function of threshold at zero, it is really cheap to compute and it’s also easy to compute the derivative in the backward propagation. 

$$
\frac{d}{dx} \, \text{ReLU}(x) = 
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$


### Fully Connected Layer

While convolutional layers are good at extracting features, it cannot directly give us the final result. For example, we want to know the probability of each class in a classification task. Fully connected layers here serve an important role in transforming learned features into a final prediction, not exclusively on classification. See MLP chapter for more details of the FC layer.  


## Backpropagation in CNNs

It is intimidating to do backpropagation in CNN when you first see it because it’s hard to imagine how to do backpropagation through multiple convolutional kernels and pooling layers. The good news is it shares the same idea as backpropagation in MLP! It computes gradients of the loss function with respect to all trainable parameters by applying the chain rule of calculus in reverse. 


### Convolutional Layer Backprop

In the forward path of the convolutional layer, we slide the filter over the input data and compute the dot product of the filter and the input (see previous section for more details). Formally, we can denote the loss function as L, input as X, filter as F,  output as O, and the convolution function as f(X, F). In this section, we assume stride is 1 for simplicity.  

$$ O_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} F_{m,n} \cdot X_{i+m,\,j+n} $$

In the backprop, we aim to find dL/dF as F has learnable parameters and we want to update the filter at each step. Similar to the backprop process in MLP, we need to have the gradient from the l + 1 layer and the gradient of the convolution function to compute the gradient of F at the l layer. By using chain rule, we have:

$$ \frac{dL}{dF} = \frac{dL}{dO} \cdot \frac{dO}{dF} $$

As dL/dO is given, we just need to calculate dO_{i, j}/dF_{p, q}. The only term in the sum that depends on F_{p, q} is $F_{p, q} * X_{i + p, j + q}$


All other filter elements F_{m, n} are multiplied with different parts of X,, and are unaffected when computing the derivative with respect to F_{p, q}, so we can have the derivative of O w.r.t F is:

$$ \frac{\partial O_{i, j}}{\partial F_{p, q}} = X_{i + p, j + q} $$


Therefore, by applying chain rule, we can compute how the loss L changes with respect to each filter weight as:

$$
\frac{\partial L}{\partial F_{p,q}} = \sum_{i,j} \frac{\partial L}{\partial O_{i,j}} \cdot \frac{\partial O_{i,j}}{\partial F_{p,q}} 
= \sum_{i,j} \delta_{i,j} \cdot X_{i+p,\,j+q}
$$

As deep CNN usually has more than one layer, we need to propagate the derivative through each layer using dL/dX. It is similar to how we compute dL/dF. Firstly, we use chain rule again to get dL/dX: 

$$ \frac{dL}{dX} = \frac{dL}{dO} \cdot \frac{dO}{dX} $$


Since the convolution uses a sliding window, each pixel X_{p, q} appears in multiple output patches. So we must sum all parts of the output that depended on X_{p, q}:

$$ \frac{\partial L}{\partial X_{p,q}} = \sum_{i,j} \frac{\partial L}{\partial O_{i,j}} \cdot \frac{\partial O_{i,j}}{\partial X_{p,q}} $$

Now we need to compute dO/dX for each X. For every O_{i, j}, X_{p, q} appears in the output only if (p, q) = (i + m, j+ n), where (m, n) is the index of the filter.

$$ \frac{\partial O_{i,j}}{\partial X_{p,q}} = F_{m,n} \quad \text{where} \quad m = p - i,\; n = q - j $$

Therefore, we can compute the gradient w.r.t input X as:

$$ \frac{\partial L}{\partial X_{p,q}} = \sum_{(i,j)\;\text{s.t.}\; X_{p,q} \in \text{patch}_{i,j}} \frac{\partial O_{i,j}}{\partial F_{p,q}} \cdot F_{p - i,\, q - j} $$

Let’s go through an example with an input of size 3*3, filter of size 2*2, and with stride = 1 and padding = 0.


### Pooling Layer Backprop

In this section, we will focus on backpropagation in the max-pooling layer. Since max-pooling doesn’t have any weights, we don’t need to update the max-pooling layer and dL/dX is all you need. By using chain rule, we have the equation for computing the derivative of L w.r.t X as:

$$ \frac{\partial L}{\partial X_{p,q}} = \sum_{i,j} \frac{\partial L}{\partial O_{i,j}} \cdot \frac{\partial O_{i,j}}{\partial X_{p,q}} $$

However, the gradient of the loss function w.r.t. the input feature map X is nonzero only if X_{i, j} is the maximum element of the kernel window. Mathematically, it means:

$$ \frac{\partial Y_{i,j}}{\partial X_{p,q}} = 
\begin{cases}
1 & \text{if } X_{p,q} = Y_{i,j} \; \text{(i.e., it was the max)} \\
0 & \text{otherwise}
\end{cases} $$


Intuitively, you can understand it as max-pooling layer selects the maximum element of the kernel window and ignores the other elements, so y is independent of x_j, where j != i, suppose y = x_i = max(x_0 … x_n). Hence, the gradient of y w.r.t. x_i should be 1 and all other elements should be 0. Therefore, we can conclude that the gradient of the max-pooling layer is:

$$ \frac{\partial L}{\partial X_{p,q}} = 
\sum_{\substack{i,j \;\text{s.t.} \\ X_{p,q} \in patch_{i, j}}}
 \frac{\partial L}{\partial O_{i,j}} \cdot \mathbf{1}\left(X_{p,q} = O_{i,j}\right) $$


## Visualizing What CNN Learns



## Transfer Learning & Fine-Tuning

- **Transfer Learning**: Use a pretrained CNN as a feature extractor.
- **Fine-Tuning**: Continue training the pretrained model on your specific dataset.

## Applications

- **Image Classification**
- **Object Detection**

## References

1. Silver, D. et al. *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*. [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)  
2. Stanford CS231n: [Convolutional Networks Guide](https://cs231n.github.io/convolutional-networks/)  
3. [AlphaZero paper (PDF)](https://arxiv.org/pdf/1511.08458)  
4. [Lecun98 CNN paper (PDF)](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)  
5. [Understanding CNNs](https://cs231n.github.io/understanding-cnn/)  
6. [Transfer Learning Notes](https://cs231n.github.io/transfer-learning/)