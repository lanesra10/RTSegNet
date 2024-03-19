# RTSeg-Net: A Lightweight Network on the Edge Computing Platform for Real-time Segmentation of Fetal Head and Pubic Symphysis from Intrapartum Ultrasound Images

Official pytorch code for "RTSeg-Net: A Lightweight Network on the Edge Computing Platform for Real-time Segmentation of Fetal Head and Pubic Symphysis from Intrapartum Ultrasound Images"

- [ ] Code release
- [ ] Paper release

## Abstract
The challenges for automatic segmentation algorithm of the fetal head (FH) and pubic symphysis (PS) in fetal descent monitoring arise from variable image quality, requirement of timeliness and limited computing resources available on wireless low-energy ultrasound devices. To overcome these challenges, we propose a lightweight deep learning network called RTSeg-Net, which can be deployed on an edge computing platform. It incorporates 1) distribution shifting convolutional blocks to reduce the number of network parameters and thus computational complexity, 2) tokenized multilayer perceptron blocks to learn a non-linear transformation of the input image and to extract extended context information, and 3) efficient feature fusion blocks with a built-in channel attention mechanism to improve the effective receptive field and to fuse relevant channel features. This RTSeg-Net with 1.86M parameters has been tested with two datasets in JNU-IFM (of 3743 images) and MICCAI 2020 (of 313 images). The mean values of Jaccard index, Dice score, and average surface distance were 86.89±0.91%, 92.42±0.60%, and 2.59±0.25 on JNU-IFM and 84.92±0.67%, 90.38±0.37%, and 3.62±0.09 on MICCAI 2020, respectively. It levels state-of-the-art networks such as DBSN and TransUNet on segmentation accuracy with only 6% of hyperparameters but at a 7-fold faster inference speed of 31.13 frames per second on Jetson Nano, a Nvidia device of the lowest computing power. This suggests that RTSeg-Net can provide under-trained professionals with accurate and real-time segmentation, making it suitable for non-stop, on-site and on-demand ultrasound image analysis on robust, compact, mobile, high-resolution real-time ultrasound machines.

### RTSeg-Net:

![framework](imgs/RTSeg-Net.tif)

## Performance Comparison

<img src="imgs/performance.tif" title="preformance" style="zoom:8%;" align="left"/>


## Environment

- GPU: NVIDIA GeForce RTX3090 GPU
- Pytorch: 1.10.0 cuda 11.4
- cudatoolkit: 11.3.1



