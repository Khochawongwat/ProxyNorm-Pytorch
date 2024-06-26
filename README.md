This repository contains the implementation of Proxy Normalization by Labatie et al., in the "Proxy-Normalizing Activations to Match Batch Normalization while Removing Batch Dependence" in PyTorch, including an added 1D normalization layer. It includes two classes: `ProxyNorm2d` and `ProxyNorm1d`. These can be used in the same way as classical normalization layers like BatchNorm or LayerNorm that we all know and love. The unique feature of these layers is that they normalize post-activations using a proxy distribution, which allows them to match or even exceed the performance of traditional layers while removing batch dependence.


**DISCLAIMER**
Please note that I am not the original inventor of this technique, nor have I collaborated with the authors. My contribution is an implementation of the Proxy Normalization technique in PyTorch, with minor modifications, and the addition of a 1D normalization layer.
