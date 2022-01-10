"""
Преобразовываю датасет с автомобилями (http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
к более удобному виду для работы в torch
"""


import scipy.io as sci
import numpy as np
import os
import shutil


if not os.path.isdir('RefactoredCars'):
    os.mkdir('RefactoredCars')


# Читаем анотации к классам
cars_meta = sci.loadmat(os.path.join('Annotations', 'cars_meta.mat'))
class_names = cars_meta['class_names']
class_names = class_names.reshape(-1)
num_classes = len(class_names)
print('Прочитано ', num_classes, ' классов')

indexToNameClass = []

for i in range(num_classes):
  
    if '/' in class_names[i][0]: # Убираем всякую гадость из названий
        class_names[i][0] = class_names[i][0].replace('/', '')

    indexToNameClass.append(class_names[i][0])

print('Первые 5 анотаций к классам')
for i in range(5):

    print(indexToNameClass[i])


# Читаем анотации к train
cars_train_annos = sci.loadmat(os.path.join('Annotations', 'cars_train_annos.mat'))

annotations = cars_train_annos['annotations']
annotations = annotations.reshape(-1)
num_images = len(annotations)
print('\nПрочитано ', num_images, ' анотаций к картинкам train')


class_labels = np.empty(num_images, dtype=np.int16)
fnames = np.empty(num_images, dtype='<U9')

for i in range(num_images):

    class_labels[i] = annotations[i][4][0][0]
    fnames[i] = annotations[i][5][0]


print('Первые пять анотаций из train (номер класса и путь к картинке)')
for i in range(5):
    print(class_labels[i], fnames[i])


# Создаём директории для всех классов train

pathToTrainImages = os.path.join('RefactoredCars', 'TrainImages')
if not os.path.isdir(pathToTrainImages):
    os.mkdir(pathToTrainImages)

for nameDir in indexToNameClass:

    dirPath = os.path.join(pathToTrainImages, nameDir)

    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)

print('\nСоздано ', len(os.listdir(pathToTrainImages)),' директорий для классов train (должно быть ', num_classes, ')')


# Переносим train картинки в новые директории

print('\nНачинаю перенос картинок train в новые директории')

for i in range(num_images):

    indxClass = class_labels[i]
    fname = fnames[i]
    sourcePath = os.path.join('Images', 'cars_train', fname)
    destinationPath = os.path.join(pathToTrainImages, indexToNameClass[indxClass - 1], fname)
    if not os.path.isfile(destinationPath):
        shutil.copyfile(sourcePath, destinationPath)

    if i % 500 == 0:
        print('Перенесено ', i, ' картинок')

print('Перенос train завершён')
print('-------------------------------------')
###############################################################################################################



# Читаем анотации к test и делим их на test и validation
cars_test_annos = sci.loadmat(os.path.join('Annotations', 'cars_test_annos.mat'))

annotations = cars_test_annos['annotations']
annotations = annotations.reshape(-1)
num_images = len(annotations)
num_valid_images = num_images - num_images // 2
num_test_images = num_images - num_valid_images
print('\n\nПрочитано ', num_images, ' анотаций к картинкам test и validation')
print(num_valid_images, ' для validation')
print(num_test_images, ' для test')


class_labels_valid = np.empty(num_valid_images, dtype=np.int16)
fnames_valid = np.empty(num_valid_images, dtype='<U9')

class_labels_test = np.empty(num_images - num_valid_images, dtype=np.int16)
fnames_test = np.empty(num_images - num_valid_images, dtype='<U9')

for i in range(0, num_valid_images):

    class_labels_valid[i] = annotations[i][4][0][0]
    fnames_valid[i] = annotations[i][5][0]


print('\nПервые пять анотаций из validation (номер класса и путь к картинке)')
for i in range(5):
    print(class_labels_valid[i], fnames_valid[i])


for j, i in enumerate(range(num_valid_images, num_images)):

    class_labels_test[j] = annotations[i][4][0][0]
    fnames_test[j] = annotations[i][5][0]


print('\nПервые пять анотаций из test (номер класса и путь к картинке)')
for i in range(5):
    print(class_labels_test[i], fnames_test[i])



# Создаём директории для всех классов test
pathToTestImages = os.path.join('RefactoredCars', 'TestImages')

if not os.path.isdir(pathToTestImages):
    os.mkdir(pathToTestImages)

for nameDir in indexToNameClass:

    dirPath = os.path.join(pathToTestImages, nameDir)

    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)

print('\nСоздано ', len(os.listdir(pathToTestImages)),' директорий для классов test (должно быть ', num_classes, ')')


# Переносим test картинки в новые директории

print('\nНачинаю перенос картинок test в новые директории')

for i in range(num_test_images):

    indxClass = class_labels_test[i]
    fname = fnames_test[i]
    sourcePath = os.path.join('Images', 'cars_test', fname)
    destinationPath = os.path.join(pathToTestImages, indexToNameClass[indxClass - 1], fname)
    if not os.path.isfile(destinationPath):
        shutil.copyfile(sourcePath, destinationPath)

    if i % 500 == 0:
        print('Перенесено ', i, ' картинок')

print('Перенос test завершён')
print('-------------------------------------')


# Создаём директории для всех классов valiadtion
pathToValidImages = os.path.join('RefactoredCars', 'ValidImages')

if not os.path.isdir(pathToValidImages):
    os.mkdir(pathToValidImages)

for nameDir in indexToNameClass:

    dirPath = os.path.join(pathToValidImages, nameDir)

    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)

print('\nСоздано ', len(os.listdir(pathToValidImages)),' директорий для классов validation (должно быть ', num_classes, ')')


# Переносим validation картинки в новые директории

print('\nНачинаю перенос картинок validation в новые директории')

for i in range(num_test_images):

    indxClass = class_labels_valid[i]
    fname = fnames_valid[i]
    sourcePath = os.path.join('Images', 'cars_test', fname)
    destinationPath = os.path.join(pathToValidImages, indexToNameClass[indxClass - 1], fname)
    if not os.path.isfile(destinationPath):
        shutil.copyfile(sourcePath, destinationPath)

    if i % 500 == 0:
        print('Перенесено ', i, ' картинок')

print('Перенос validation завершён')
print('-------------------------------------')