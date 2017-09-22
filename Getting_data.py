#testing CIFAR images

import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os.path

import requests
import shutil



def download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

    return local_filename

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict




def creating_dataset():
    if  os.path.isfile('cifar-10-python.tar.gz'):
        pass

    else:
        print('**********Downloading file***************************')
        download_file('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
        print('FINISHED DOWNLOADING')




    data_files = ['C:\\Users\Moondra\\Desktop\\CIFAR_DATA\\cifar-10-batches-py\\data_batch_1',
               'C:\\Users\Moondra\Desktop\\CIFAR_DATA\cifar-10-batches-py\\data_batch_2',
             'C:\\Users\Moondra\Desktop\\CIFAR_DATA\\cifar-10-batches-py\\data_batch_3',
             'C:\\Users\Moondra\Desktop\\CIFAR_DATA\\cifar-10-batches-py\data_batch_4',
             'C:\\Users\Moondra\Desktop\\CIFAR_DATA\\cifar-10-batches-py\\data_batch_5'
             ]

    data = []
    for file  in (data_files):
        data.append(unpickle(file)) #


     #create labels and data np.arrays   
    labels = np.array(data[0][b'labels']).reshape(10000,1)
    for file in data[1:]:
        labels = np.vstack((labels, np.array(file[b'labels']).reshape(10000,1)))
    labels =np.eye(10)[labels].reshape(50000,10)
    print('labels shape: ',labels.shape)

    training_data = data[0][b'data']
    for file in data[1:]:
        training_data = np.vstack((training_data, np.array(file[b'data'])))

   


    im_r = training_data[:,:1024].reshape(50000,32,32)  #or np.newaxis
    im_g = training_data[:,1024:2048].reshape(50000,32,32)
    im_b = training_data[:,2048:].reshape(50000,32,32)

    training_data= np.concatenate((im_r [...,np.newaxis],im_g [...,np.newaxis],im_b [...,np.newaxis]), axis = -1)

    print('training data shape: ' ,training_data.shape)


    return training_data, labels

def display_cifar( images, size_dimensions):


    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()



    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size_dimensions[0])])
                    for i in range(size_dimensions[1])])

    #m2 = np.hstack(images[np.random.choice(n)] for i in range(size))

    plt.imshow(im)
    plt.show()

#Let's test plotting an image

if __name__ == '__main__':
    training_data, labels = creating_dataset()

    plt.imshow(training_data[5])
    plt.show()

    
                               
                                                       

    



               



#Be careful with backslash removing the C\\   colon. 

