accuracy_file = open('/mlops/accuracy.txt.txt','r')
input_file = open('/mlops/input.txt','r')
data_file = open('/mlops/data.txt','r')



new_accuracy = float(accuracy_file.read()) 

data = data_file.read()
data = data.split('\n')
convolve_layers = int(data[0])
no_of_filters  = int(data[1])
nfilters = int(data[2])
filter_size = int(data[3])
pool_size = int(data[4])
input_size = str(data[5])
epochs = int(data[6])
no_of_neurons = int(data[7])


inputs = input_file.read()
inputs = inputs.split('\n')


if( new_accuracy < 98.5 ):
    if(convolve_layers == 1):
              nfilters = no_of_filters * 2
    elif(convolve_layers == 3):
              no_of_neurons = no_of_neurons + 200
    else:
            nfilters = nfilters * 2
    convolve_layers = convolve_layers + 1 
    epochs = epochs + 1   


data_file.close()
input_file.close()

input_file = open('/mlops/input.txt','w')
input_file.write(str(convolve_layers ))
input_file.write('\n')
input_file.write(str(no_of_filters))
input_file.write('\n')
input_file.write(str(filter_size))
input_file.write('\n')
input_file.write(str(pool_size))
input_file.write('\n')
input_file.write(str(epochs))
input_file.write('\n')
input_file.write(str(nfilters ))
input_file.write('\n')
input_file.write(str(no_of_neurons ))
input_file.close() 