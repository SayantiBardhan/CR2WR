# CR2WR
This code is used for **co-saliency detection in images**. Here, we have set the folders for a benchmark co-saliency datasets: iCoSeg. However, any cosaliency datasets can be tested on the cod.

**Usage:**
1. In ./first_step_sal_output/, we store single image saliency detection maps generated by any state-of-the-art saliency detection method as an input for further processing. Here, we use saliency maps generated by:
H. Peng, B. Li, R. Ji, W. Hu, W. Xiong, C. Lang (2017) "Salient object detection via low-rank and structured sparse matrix decomposition", IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(4), pp. 818 – 832.

2. In ./images_iCoseg/, we store input co-saliency images.
3. In ./result_iCoseg/, we store output co-saliency maps.
4. For Testing: To test run: icoseg_rw.m 


**Dependencies:**
1. MATLAB 2016 

**If you think this work is helpful, please cite:**
