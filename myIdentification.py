import keras
import cv2
import os
import keras.layers
import keras.utils
import sklearn
import numpy as np


#########################################################
########################Configration#####################
train_path="sample/train"
test_path="sample/test"
valid_path="sample/validation"
#train\0cya_16378348962584395.png

height = 30
width = 92
n_len = 4
input_shape = (30, 92, 3)  # 根据验证码图片尺寸调整
num_classes = 36  # 验证码字符种类数
batch_size = 512
epochs = 10


pre_train_total_num=5000

############################################################
import string
characters = string.digits + string.ascii_lowercase

# 读取训练集和测试集
def read_img(path):
    img_list = []
    cnt=0
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename)) 
        cnt+=1
        if cnt>pre_train_total_num:
            break
        # print(img.shape)
        img_list.append(img)
    return img_list

#训练集和测试集
train_images=read_img(train_path)
valid_images=read_img(valid_path)
test_images=read_img(test_path)


# train_label is (n_len, batch_size, n_class)
#one-hot编码

def one_hot_encode(path):
    ans_label=[]
    cnt=0
    for filename in os.listdir(path):
        cnt+=1
        if cnt>pre_train_total_num:
            break
        filename=filename[:4]
        label=[[0 for i in range(num_classes)]
                for j in range(len(filename))]

        for i in range(len(filename)):
            label[i][characters.index(filename[i])]=1
        


        ans_label.append(label)
    return ans_label

class CaptchaSequence(keras.utils.Sequence):
    def __init__(self,img_dir,characters,batch_size,steps,n_len=4,width=170,height=80):
        self.characters=characters
        self.batch_size=batch_size
        self.steps=steps
        self.n_len=n_len
        self.width=width
        self.height=height
        self.n_class=len(characters)

        self.img_paths = []  # 存储所有图片路径
        self.labels = []  # 存储对应标签
        # 遍历目录，加载图片路径和标签
        for filename in os.listdir(img_dir):
            img_path = img_dir +'/' + filename  # 图片路径
            # 假设标签是文件名去掉扩展名
            label = filename[:4]  
            if len(label) == n_len and all(c in characters for c in label):
                self.img_paths.append(img_path)
                self.labels.append(label)
        
    def __len__(self):
        return self.steps
    def __getitem__(self,idx):
        x=np.zeros((self.batch_size,self.height,self.width,3),dtype=np.float32)
        #初始化图片数据全是0，按固定输入维度
        y=[np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        #初始化4个输出[4个输出层，batch_size,n_classes]
        for i in range(self.batch_size):
            if idx * self.batch_size + i >= len(self.img_paths):
                idx = 0  # 或者其他处理方式以避免重复
            img_path = self.img_paths[idx * self.batch_size + i]
            label = self.labels[idx * self.batch_size + i]
            # 加载图片并调整尺寸
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.width, self.height))
            x[i] = img.astype(np.float32) / 255.0  # 归一化
            # 处理标签
            for j, ch in enumerate(label):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return x,y


train_labels=np.array(one_hot_encode(train_path),dtype=np.float32)
valid_labels=np.array(one_hot_encode(valid_path),dtype=np.float32)
test_labels=np.array(one_hot_encode(test_path),dtype=np.float32)



print(train_labels[0])
print(test_labels[0])



# exit(0)

# 数据预处理

from keras.utils import np_utils

# train_label is (n_len, batch_size, n_class)
# 取出其中概率最大的四个字符

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])


# 归一化
train_images = np.array(train_images) / 255.0
valid_images = np.array(valid_images) / 255.0
test_images = np.array(test_images) / 255.0





# 数据生成器
# from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     fill_mode='nearest')


# train_generator = datagen.flow_from_directory(
#     valid_path,  # 训练数据目录
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode='sparse')

# validation_generator = datagen.flow_from_directory(
#     valid_path,  # 验证数据目录
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode='sparse')

#自定义数据生成器
# def custom_data_generator(data, labels, batch_size):
#     while True:  # 无限循环，以便于持续生成数据
#         for i in range(batch_size):
#             img = data[i]
#             label = labels[i]  
#         yield img, label

# 创建自定义数据生成器
# train_generator = custom_data_generator(train_images, train_labels, batch_size)
# validation_generator = custom_data_generator(valid_images, test_labels, batch_size)
# test_generator = custom_data_generator(test_images, test_labels, batch_size)

model=keras.Sequential()

# 添加卷积层
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D((2, 2)))
for i in range(len(train_images)):
    # 添加卷积层
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))


# 添加全连接层
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

print(model.summary())


exit(0)
# 编译模型，使用Adadelta优化器，损失函数为交叉熵损失
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])




# 训练模型，使用训练集数据，标签为one-hot编码，批处理大小为32，迭代次数为10， verbose为1，验证集数据为测试集数据
# employ KFold
# from sklearn.model_selection import KFold
# n_splits=10
# kfold=KFold(n_splits=n_splits,shuffle=True,random_state=np.random.seed(7))

history = model.fit(
    train_images,train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(valid_images, valid_labels),
    verbose=1)





#######
# 评估模型
#######
# 评估模型，使用测试集数据，批处理大小为32， verbose为1
test_loss, test_acc = model.evaluate(test_images, batch_size, verbose=1)

print('Test accuracy:', test_acc)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

from matplotlib import pyplot as plt

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

print(model.summary())