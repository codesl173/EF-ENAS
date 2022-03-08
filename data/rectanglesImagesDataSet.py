import os
import pickle
import numpy as np


def extract_rectangles(dir_path):
    # Extract the zip file
    print('Extracting the dataset')
    import zipfile
    fh = open(os.path.join(dir_path, 'rectangles_images.zip'), 'rb')
    z = zipfile.ZipFile(fh)

    dir_path = os.path.join(dir_path, 'rectanglesImages')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    for name in z.namelist():
        s = name.split('/')
        outfile = open(os.path.join(dir_path, s[len(s) - 1]), 'wb')
        outfile.write(z.read(name))
        outfile.close()
    fh.close()

    train_file_path = os.path.join(dir_path, 'rectangles_im_train.amat')
    valid_file_path = os.path.join(dir_path, 'rectangles_im_valid.amat')
    test_file_path = os.path.join(dir_path, 'rectangles_im_test.amat')

    # Split data in valid file and train file
    fp = open(train_file_path)

    # Add the lines of the file into a list
    lineList = []
    for line in fp:
        lineList.append(line)
    fp.close()

    # Create valid file and train file
    valid_file = open(valid_file_path, "w")
    train_file = open(train_file_path, "w")

    # Write lines into valid file and train file
    for i, line in enumerate(lineList):
        if ((i + 1) > 10000):
            valid_file.write(line)
        else:
            train_file.write(line)

    valid_file.close()
    train_file.close()


def divide_dataset(data_np, data_length, k):
    data = data_np.reshape([data_length, -1])
    data_image = data[:, :-1].reshape([data_length, k,k])
    data_label = data[:, -1]
    data_label = np.array(data_label, dtype=np.int)
    return data_image, data_label


def load_rectangles_im(dir_path):
    data_save_path = os.path.join(dir_path, 'rectanglesImages_py.dat')
    if os.path.exists(data_save_path):
        with open(data_save_path, 'rb') as f:
            [(train_image, train_label), (valid_image, valid_label), (test_image, test_label)] = pickle.load(f)
        return (train_image, train_label), (valid_image, valid_label), (test_image, test_label)
    train_file_path = os.path.join(dir_path, 'rectangles_im_train.amat')
    valid_file_path = os.path.join(dir_path, 'rectangles_im_valid.amat')
    test_file_path = os.path.join(dir_path, 'rectangles_im_test.amat')
    assert os.path.exists(train_file_path)
    assert os.path.exists(valid_file_path)
    assert os.path.exists(test_file_path)

    # read data set
    with open(os.path.expanduser(train_file_path)) as f:
        string = f.read()
        train_data = np.array([float(i) for i in string.split()])
    train_image, train_label = divide_dataset(train_data, 10000, 28)
    with open(os.path.expanduser(valid_file_path)) as f:
        string = f.read()
        valid_data = np.array([float(i) for i in string.split()])
    valid_image, valid_label = divide_dataset(valid_data, 2000, 28)
    with open(os.path.expanduser(test_file_path)) as f:
        string = f.read()
        test_data = np.array([float(i) for i in string.split()])
    test_image, test_label = divide_dataset(test_data, 50000, 28)

    # save data file
    with open(data_save_path, 'wb') as f:
        pickle.dump([(train_image, train_label), (valid_image, valid_label), (test_image, test_label)], f)

    return (train_image, train_label), (valid_image, valid_label), (test_image, test_label)


if __name__ == '__main__':
    rectangles_path = r'./'
    extract_rectangles(rectangles_path)
    # (train_i, train_l), (valid_i, valid_l), (test_i, test_l) = \
    #     load_rectangles_im(os.path.join(rectangles_path, 'rectanglesImages'))

