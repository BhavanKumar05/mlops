from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils
import keras

# loads the MNIST dataset
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

# Lets store the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# Getting our date in the right 'shape' needed for Keras
# We need to add a 4th dimenion to our date thereby changing our
# Our original image shape of (60000,28,28) to (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# store the shape of a single image 
input_shape = (img_rows, img_cols, 1)

# change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255

# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

input_file = open('/mlops/input.txt','r')
inputs = input_file.read()
inputs = inputs.split('\n')
convolve_layers = int(inputs[0])
nfilter = int(inputs[1])
filter_size = int(inputs[2])
pool_size = int(inputs[3])
nfilters = int(inputs[5])
no_of_neurons = int(inputs[6])


data = 'No. of convolve layers : ' + str(convolve_layers) + '\nLayer 1' +  '\nNo of filters : ' + str(nfilter) + '\nFilter Size :  (' + str(filter_size) + ',' +str(filter_size) + ')' + '\nPool Size : ( ' + str(pool_size) + ',' +str(pool_size) + ')' 







# create model
model = Sequential()

# 2 sets of CRP (Convolution, RELU, Pooling)
model.add(Conv2D(nfilter, (filter_size, filter_size),
                 padding = "same", 
                 input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (pool_size, pool_size)))

#Subsequent CRP sets
for i in range(1,convolve_layers):
	nfilters = nfilters 
	filter_size = int(inputs[2])
	pool_size =  int(inputs[3])
	data = data + '\nLayer ' + str(i+1) + ': '
	data = data + '\nNo of filters : ' + str(nfilters) + '\nFilter Size : ( '  + str(filter_size) + ','+ str(filter_size) + ')' + '\nPool Size : ( ' + str(pool_size) + ',' +str(pool_size) + ')'
	model.add(Conv2D(nfilters, (filter_size, filter_size),padding = "same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size = (pool_size, pool_size)))
# Fully connected layers (w/ RELU)
model.add(Flatten())
model.add(Dense(no_of_neurons))
model.add(Activation("relu"))

# Softmax (for classification)
model.add(Dense(num_classes))
model.add(Activation("softmax"))
           
model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])
    
print(model.summary())


# Training Parameters
batch_size = 128
epochs = int(inputs[4])
data = data  + '\nepochs : ' + str(epochs)

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model.save("mnist_LeNet.h5")

# Evaluate the performance of our trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


accuracy_file = open('/mlops/accuracy.txt.txt','w')
accuracy_file.write( str(scores[1]))
accuracy_file.close()


data_file = open('/mlops/data.txt','w')
data_file.write(str(convolve_layers))
data_file.write('\n')
data_file.write(str(nfilter))
data_file.write('\n')
data_file.write(str(nfilters))
data_file.write('\n')
data_file.write(str(filter_size))
data_file.write('\n')
data_file.write(str(pool_size))
data_file.write('\n')
data_file.write(str(input_shape))
data_file.write('\n')
data_file.write(str(epochs))
data_file.write('\n')
data_file.write(str(no_of_neurons))
data_file.close()



display_file = open('/mlops/data_per_layer.html','r+')
display_file.read()
display_file.write('<pre>\n*************************************************\n')
display_file.write(data)
display_file.write('\nAccuracy achieved : ' + str(scores[1])+'\n</pre>')
display_file.close()

