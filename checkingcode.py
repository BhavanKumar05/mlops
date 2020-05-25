programfile = open('/root/mlops/lenetprogram.py','r')
code = programfile.read()				

if 'keras' or 'tensorflow' in code:		
	if 'Conv2D' or 'Convolution' in code:				
		print('lenetCNN')
	else:
		print('CNN')
else:
	print('not deep learning')
