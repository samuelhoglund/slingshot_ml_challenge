# Circle Detector
The following code contains a PyTorch-built CNN model trained to identify the center coordinates and radius of a circle in a 100x100 greyscale image with arbitrary noise ranging from 0 to 1. The model weights are saved in the file `circle_model.pth`. The code is formatted using the Black formatter. Below, I describe each section of the Jupyter notebook `circle_detector.ipynb`.

## The model
The network consists of approximately 8.2M parameters, divided into five convolutional layers and three fully connected layers. I added the method `_forward_conv()` to help calculate the dimensions of the convolutional layers to correctly set the input features of the first fully connected layer. This is calculated in the method `_calculate_conv_output_size()` that takes the image dimensions as arguments.

## The dataset class
This class handles the dataset and makes it compatible with the PyTorch dataloader module. The class has the attributes `num_samples`, `noise_level`, `img_size`, `min_radius`, and `max_radius` that can be set freely. In the notebook, the noise level and image sizes are defined as static variables in a code snippet. I also create a `generator` through the `generate_examples` function in the given code. The class  implements the `__len__` and `__getitem__` methods required for integration with PyTorch's dataloader. By specifying the number of samples and noise level, you can create an instance of the CircleDataset class used to generate data for training.

## The training loop
In the training loop, I start by defining the training hyperparameters, generating a dataset and defining the loss function and optimizer. I use the `Adam` optimization algorithm due to its popularity and proven efficiency. The training data consisted of 120,000 samples trained in 10 epochs with a batch size of 32. 

The final model was trained on a dataset with images that had their noise level set to 0.6, which showed a high performance of identifying images with noise levels in the range [0, 0.6], then a worsened performance for higher noise levels. I noticed that training a model on a higher noise level generally only resulted in decent performance for that specific level and low performance for the other levels (with IoU scores around or below 0.5). 

## Evaluation
For evaluating the model, I test it on a dataset of 1,100 samples across 11 different noise levels from 0 to 1, with 100 samples of each noise level. I then calculate the average IoU for each noise level. The `img_tensor` is given its dimensions to match the input shape that the model expects. 

In the code, it is possible to load the model weights from the `circle_model.pth` file.

The performance of the final model is the following:
```
Average IoU for noise level 0: 0.9260975771661327
Average IoU for noise level 0.1: 0.9248244487549916
Average IoU for noise level 0.2: 0.9307616789817341
Average IoU for noise level 0.3: 0.9279325732484223
Average IoU for noise level 0.4: 0.9321503261895749
Average IoU for noise level 0.5: 0.9303383664376723
Average IoU for noise level 0.6: 0.8653112519508572
Average IoU for noise level 0.7: 0.545983215936083
Average IoU for noise level 0.8: 0.2209339611156755
Average IoU for noise level 0.9: 0.1359130410140969
Average IoU for noise level 1: 0.09061128967740162
```
