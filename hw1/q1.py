import scipy.io
from sklearn import svm
trainset_size = [100, 200, 500, 1000, 2000, 5000]
digit_data_test = scipy.io.loadmat("data/digit-dataset/test.mat")
digit_data_train = scipy.io.loadmat("data/digit-dataset/train.mat")

test_img= digit_data_test['test_images']
train_img= digit_data_train['train_images']
train_label= digit_data_train['train_labels']
train=[]
for i in np.arange(shape(train_img)[2]):
    train.append(train_img[:,:,1].flatten())
train= np.array(train)
clf = svm.LinearSVC()
clf.fit(train[100:], train_label[100:])  
