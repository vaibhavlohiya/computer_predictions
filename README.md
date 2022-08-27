# Computer Predictions
Making an artificial intelligence model to differentiate between the photos of cats and dogs using convolution neural networks (CNN) and artificial neural networks (ANN).

This simple process in python can be divided into 4 stages.

1. Pre-processing
2. Building the neural network
3. Training the neural network
4. Making a prediction.

## Pre-processing 
At the beginenning we would need two important modules to make our neural network. First, we need the `tensorflow` module. If you don't have it you can simply download it by typing `pip install tensorflow` on your command prompt. Second, we need the `keras` module to use it's famous `ImageDataGenerator` method.

In pre-processing, the main job is to make the training and the test readable for our neural network to make accurate predictions. Our network cannot simply look at the image and say that 'It is a cat' or 'It is a bike'. We use our ImageDataGenerator to convert the data of our image into somewhat more readable for our neural network (more like a spreadsheet where each block is a single pixel and a number is written, depicting the intenstiy of RGB colours in that pixel)

Now, after converting all the training set and test set images by our method, we are pretty much done with pre-processing.

### Building the neural network
We first initialize the network by using the following command `cnn = tf.keras.models.Sequential()` and when the CNN (Convolution Neural Network) is created we move forward by following these steps.

1. Convolution - Adding the parameters of `Conv2D` to the cnn.
  
   `cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64, 64, 3]))`

2. Pooling - `cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))`

3. Flattening - `cnn.add(tf.keras.layers.Flatten())`

4. Full Connection - 

   `cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))`

5. Output layer - For the output layer we can choose our activation function. Here, we chose **sigmoid**. 

   `cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))`
   
 ## 3. Training the CNN
 Training the network is fairly simple we just compile our output layer and then feed our training-set to the CNN by entering the number of `epochs` in which the CNN will divide the training set.
 
 `cnn.fit(x = training_set, epochs = 25)`
 
 ## 4. Making a prediction
 Now, after looking at over 8000 photos of cats & dogs. Our network must know by now that which one's which. To check we feed a dog image to our 'trained network' and see if it can make correct predictions. We import `image` method from `keras.preprocessing` to read the prediction image and `numpy` to expand the image's dimensions.
 Lastly, we put a mark to show if our network predicts cat or dog respectively.
