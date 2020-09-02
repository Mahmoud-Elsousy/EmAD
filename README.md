# EmAD
This repository is hosting EmAD -Embedded Anomaly detection using Machine Learning-  which was build for my Master's project at DCU. EmAD is a framework to allow fast training and deployment of anomaly detection algorithms on ARM32 embedded devices running on Linux. EmAD is packaged as a docker image for ARM32 devices and uploaded on Docker Hub, it was tested on Raspberry Pi4 and Beaglebone AI. EmAD can be used on x86/x64 systems using a QEMU privileged container on Docker using the following commands:

```bash
docker run --rm --privileged hypriot/qemu-register
docker run -it --name EmAD_Dash -p 4444:4444 emad-dash-image
```

#### 1. Anomaly Detection Algorithms available in EmAD:

All the available algorithms in EmAD depend on [PyOD](https://pyod.readthedocs.io/en/latest/index.html)'s implementation, the following table lists all the available algorithms in EmAD, this is a subset of the implemented algorithms in PyOD.

|    | Abbreviation |               Full name               |
|----|:------------:|:-------------------------------------:|
| 1  | PCA          | Principal Component Analysis          |
| 2  | MCD          | Minimum Covariance Determinant        |
| 3  | OCSVM        | One-Class Support Vector Machines     |
| 4  | LMDD         | Deviation-based Outlier Detection     |
| 5  | LOF          | Local Outlier Factor                  |
| 6  | COF          | Connectivity-Based Outlier Factor     |
| 7  | CBLOF        | Clustering-Based Local Outlier Factor |
| 8  | HBOS         | Histogram-based Outlier Score         |
| 9  | kNN          | k Nearest Neighbors                   |
| 10 | SOD          | Subspace Outlier Detection            |
| 11 | ABOD         | Angle-Based Outlier Detection         |
| 12 | IForest      | Isolation Forest                      |
| 13 | FB           | Feature Bagging                       |

#### 2. EmAD Interface Images

![data-page](/Interface-images/00-data-page.png)
![training-page](/Interface-images/01-training-page.png)
![testing-page](/Interface-images/02-testing-page.png)
![deployment-page](/Interface-images/03-deployment-page.png)