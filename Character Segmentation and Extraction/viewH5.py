import h5py
import matplotlib.pyplot as plt
import numpy as np

def print_hdf5_item(name, obj):
    if isinstance(obj, h5py.Group):
        print("Group:", name)
    elif isinstance(obj, h5py.Dataset):
        print("Dataset:", name, "   Shape:", obj.shape, "   Dtype:", obj.dtype)
        for each in obj:
            print(each)
    else:
        print("Unknown item type:", name)

    # Print attributes
    for attr_name, attr_value in obj.attrs.items():
        print("   Attribute:", attr_name, "=", attr_value)


def viewH5(filePath):
    with h5py.File(filePath, 'r') as hf:
        load_arr = hf['dataset'][:]

    class_names = ['Stefan', 'Johan', 'David', 'StefanP', 'Brian', 'Keegan', 'Myburgh']
    imagesToShow = []
    for image in load_arr:
        imagesToShow.append(np.reshape(image[1:], (35, 35, 1)))

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imagesToShow[i])
    plt.show()