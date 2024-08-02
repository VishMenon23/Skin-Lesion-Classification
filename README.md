# Skin-Lesion-Classification

## Abstract

This project aims to enhance the accuracy of skin cancer detection using modern deep learning architectures, specifically Vision Transformer (ViT) models, and compare them against CNN-based models such as ResNet50, VGG16, DenseNet121, VGG19, and ResNet152. The transformer models explored include ViT, Convolutional Vision Transformer (CvT), BERT Pre-Training of Image Transformers (BEiT), and Hierarchical Vision Transformer using Shifted Windows (Swin Transformer). The primary dataset used is HAM10000.

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
    - [Dataset](#dataset)
    - [Dataset Augmentation](#dataset-augmentation)
    - [Normalization of Input Images](#normalization-of-input-images)
2. [Transformer Models](#transformer-models)
    - [Vision Transformer (ViT)](#vision-transformer-vit)
    - [Convolutional Vision Transformer (CvT)](#convolutional-vision-transformer-cvt)
    - [Bidirectional Encoder representation from Image Transformers (BEiT)](#bidirectional-encoder-representation-from-image-transformers-beit)
    - [Hierarchical Vision Transformer using Shifted Windows (Swin Transformer)](#hierarchical-vision-transformer-using-shifted-windows-swin-transformer)
3. [Results and Performance Analysis](#results-and-performance-analysis)
    - [Confusion Matrix](#confusion-matrix)
    - [Accuracy, Precision, Recall, and F1 Score](#accuracy-precision-recall-and-f1-score)
4. [Conclusion](#conclusion)

## Data Preprocessing

### Dataset

The HAM10000 dataset consists of 10015 dermatoscopic images representing seven types of skin lesions. The images are standardized to a 224x224x3 RGB format.

### Dataset Augmentation

Due to the imbalance in the dataset, augmentation techniques such as flipping, rotating, and zooming are applied to balance the data. After augmentation, the total number of images increases to 35967.

### Normalization of Input Images

Normalization is performed using z-score normalization based on the dataset's mean and standard deviation to ensure consistent results and faster convergence.

## Transformer Models

### Vision Transformer (ViT)

ViT splits the input image into patches and feeds the linear embeddings of these patches into the Transformer. A classification token is used to predict the class probabilities.

### Convolutional Vision Transformer (CvT)

CvT introduces convolutional operations on patch embeddings to capture spatial contexts, combining the strengths of CNNs and Transformers.

### Bidirectional Encoder representation from Image Transformers (BEiT)

BEiT uses a masked image modeling task to pretrain vision Transformers, predicting masked patches based on the encoding vectors of the corrupted image.

### Hierarchical Vision Transformer using Shifted Windows (Swin Transformer)

Swin Transformer constructs a hierarchical representation with non-overlapping windows, achieving linear computational complexity and capturing global context.

## Results and Performance Analysis

### Confusion Matrix

Confusion matrices are used to evaluate the performance of both CNN and transformer models.

### Accuracy, Precision, Recall, and F1 Score

Performance metrics such as accuracy, precision, recall, and F1 score are calculated to compare different models.

## Conclusion

The project demonstrates that Vision Transformer models, particularly the base ViT and SWIN Transformer, surpass CNN models in accuracy for skin cancer detection. Transformers are well-suited for image classification tasks due to their ability to capture global context and efficient parameter usage.

## References

1. Wu, Haiping, et al. "Cvt: Introducing convolutions to vision transformers." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
2. Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
3. Bao, Hangbo, et al. "Beit: Bert pre-training of image transformers." arXiv preprint arXiv:2106.08254 (2021).
4. Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
5. Uğur Fidan, İsmail Sarı, and Raziye Kübra Kumrular. Classification of skin lesions using ann. In 2016 Medical Technologies National Congress (TIPTEKNO), pages 1–4. IEEE, 2016.
6. Mobeen ur Rehman, Sharzil Haris Khan, SM Danish Rizvi, Zeeshan Abbas, and Adil Zafar. Classification of skin lesion by interference of segmentation and convolution neural network. In 2018 2nd International Conference on Engineering Innovation (ICEI), pages 81–85. IEEE, 2018.
7. M Monisha, Alex Suresh, BR Tapas Bapu, and MR Rashmi. Classification of malignant melanoma and benign skin lesion by using back propagation neural network and abcd rule. Cluster Computing, 22(5):12897–12907, 2019.
8. Marwan Ali Albahar. Skin lesion classification using convolutional neural network with novel regularizer. IEEE Access, 7:38306–38313, 2019.
9. Andre Esteva, Brett Kuprel, Roberto A Novoa, Justin Ko, Susan M Swetter, Helen M Blau, and Sebastian Thrun. Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639):115–118, 2017.
10. Fangfang Han, Huafeng Wang, Guopeng Zhang, Hao Han, Bowen Song, Lihong Li, William Moore, Hongbing Lu, Hong Zhao, and Zhengrong Liang. Texture feature analysis for computer-aided diagnosis on pulmonary nodules. Journal of digital imaging, 28(1):99–115, 2015.
