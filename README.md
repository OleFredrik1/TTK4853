# TTK4853 - Experts in Team - XAI

(Placeholder)

This is our XAI project for the course Experts in Team at NTNU in 2021.


### Helpful resources   
- https://github.com/Vipermdl/Grad-CAM  
- https://keras.io/examples/vision/grad_cam/  
- https://www.tensorflow.org/api_docs/python/tf/GradientTape  

### Grad-CAM  
Gradient-weighted Class Activation Mapping
(Grad-CAM), uses the gradients of any target concept (say
‘dog’ in a classification network or a sequence of words
in captioning network) flowing into the final convolutional
layer to produce a coarse localization map highlighting the
important regions in the image for predicting the concept.  
We perform a weighted combination of forward activation
maps, and follow it by a ReLU. We apply a ReLU to the linear combination
of maps because we are only interested in the features that
have a positive influence on the class of interest, i.e. pixels
whose intensity should be increased in order to increase y
c.  
*Note that Z is the number of pixels in the feature map*  
#### Steps  
* First, we create a model that maps the input image to the activations of the last conv layer  
* Second, we create a model that maps the activations of the last conv layer to the final class predictions  
* Then, we compute the gradient of the top predicted class for our input image with respect to the activations of the last conv layer  
* Get the gradient of the top predicted class with regard to the output feature map of the last conv layer  
```Python
# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the top predicted class
last_conv_layer_output = last_conv_layer_output.numpy()[0]
pooled_grads = pooled_grads.numpy()
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(last_conv_layer_output, axis=-1)
```
* Optionally normalize the heatmap afterwards  
#### Grad-CAM++  
Grad-CAM fails to properly localize objects
in an image if the image contains multiple occurrences of the
same class. This is a serious issue as multiple occurrences of
the same object in an image is a very common occurrence
in the real world. Another consequence of an unweighted
average of partial derivatives is that often, the localization
doesn’t correspond to the entire object, but bits and parts
of it. This can hamper the user’s trust in the model, and
impede Grad-CAM’s premise of making a deep CNN more
transparent.  
*Grad-CAM++ only considers positive gradients when calculating the saliency map. (I thought regular Grad-CAM did this as well?)*
