# PIV-LiteFlowNet-en

***PIV-LiteFlowNet-en*** is a deep neural network performing particle image velocimetry (PIV), which is a visualization technique for fluid motion estimation.

## Directory in this repository

>**caffe**: folder as the caffe master with the trained models  
>**demos**: folder containing MATLAB scripts for testing the trained models

<br>

## License and citation

This repository is provided for research purposes only. All rights reserved. Any commercial use requires our consent. If you use the codes in your research work, please cite the following paper: 

	Cai S, Liang J, Gao Q, Xu C, Wei R. Particle image velocimetry based on a deep learning motion estimator, submitted to IEEE Transactions on Instrumentation and Measurement. (arXiv version to be added) 
	
or the [predecessor paper](https://doi.org/10.1007/s00348-019-2717-2)

	Cai S, Zhou S, Xu C, Gao Q. Dense motion estimation of particle images via a convolutional neural network[J]. Exp Fluids, 2019, 60(4): 73.

The caffe package comes from [LiteFlowNet](https://github.com/twhui/LiteFlowNet) with our new training prototxt template as well as the models trained on PIV dataset. LiteFlowNet is proposed by Hui et al. (2018) in the following paper: 

	Hui T-W, Tang X, Loy C. Liteflownet: A lightweight convolutional neural network for optical flow estimation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

<br>

## Installation and Compiling


Installation was tested under Ubuntu 14.04/16.04 with CUDA 8.0, cuDNN 5.1. The codes of demos are tested in MATLAB R2017b. Before using the trained networks, make sure that the installation of  **[CUDA](https://developer.nvidia.com/cuda-downloads)  and  [CuDNN](https://developer.nvidia.com/cudnn)** is completed. 

Compile [caffe](http://caffe.berkeleyvision.org/) by editing "caffe/Makefile.config" (if necessary), then 

	$ cd caffe
	$ make all
	$ make matcaffe

<br>

## Trained models

The pre-trained models are available in the folder  `caffe/models/`. 
- **PIV-LiteFlowNet**: Original LiteFlowNet trained on a PIV dataset. 
- **PIV-LiteFlowNet-en**: LiteFlowNet with minor revisions trained on a PIV dataset. 

The PIV dataset can be referred to [https://github.com/shengzesnail/PIV_dataset](https://github.com/shengzesnail/PIV_dataset).

<br>

## Testing
(This assumes that you compiled the caffe code sucessfully)

1. Open the `demos` folder
2. Run `demos/test_pivLiteflownet.m` in MATLAB. 
	It is a script for evaluating PIV-LiteFlowNet-en on a single image pair. Several demos are given in the folder `demos/testedData/`. If everything goes fine, you can see the results like these: 
	
	**- Vortex pair flow**
	<figure  class="third">
	<img src="https://github.com/shengzesnail/PIV-LiteFlowNet-en/raw/master/demos/testedData/vortexPair_images.gif" width="250"/>     <img src="https://github.com/shengzesnail/PIV-LiteFlowNet-en/raw/master/demos/testedData/vortexPair_results.png" width="280"/>
	</figure>
	
	**- Backward stepping flow**
	<figure  class="third">
	<img src="https://github.com/shengzesnail/PIV-LiteFlowNet-en/raw/master/demos/testedData/backstep_images.gif" width="250"/>     <img src="https://github.com/shengzesnail/PIV-LiteFlowNet-en/raw/master/demos/testedData/backstep_results.png" width="460"/>
	</figure>
	
	**- [DNS turbulent flow](http://fluid.irisa.fr/data-eng.htm)**
	<figure  class="third">
	<img src="https://github.com/shengzesnail/PIV-LiteFlowNet-en/raw/master/demos/testedData/DNS_tur_images.gif" width="250"/>     <img src="https://github.com/shengzesnail/PIV-LiteFlowNet-en/raw/master/demos/testedData/DNS_tur_results.png" width="460"/>
	</figure>
	
3. Run `demos/test_pivLiteflownet_all.m` in MATLAB. 
	This is a script for evaluating PIV-LiteFlowNet-en on a list of images. An image sequence of uniform flow is provided in the folder `demos/testedData2/`. The root mean square error (RMSE) and the computation time are evaluated, thus you can assess the accuracy and efficiency of the CNN model. 

<br>

## Training

(This assumes that you compiled the caffe code sucessfully)

Download the [PIV dataset](https://github.com/shengzesnail/PIV_dataset) and convert to `LMDB` format. This can be referred to `caffe/data/make-lmdbs-train.sh`

The configuration files for training a neural network on Caffe (including `train.prototxt` and `solver.prototxt`) are provided in the folder `caffe/models/training_template`.    

Copy `****train.prototxt` and `solver.prototxt` to a new folder and edit the files to make sure all settings (e.g., path of your data) are correct. 

Start training ```$ caffe train --solver solver.prototxt ```

