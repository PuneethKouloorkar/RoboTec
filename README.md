# RoboTec PTC - Panoptic Segmentation

The task at hand involved adapting (implementing) the AdaptIS repository for a custom dataset. Originally, the AdaptIS repository implemented panoptic segmentation for a single class multi-instance dataset (a toy dataset) with the option to extend the code to a (custom) multi-class multi-instance dataset.

## Custom dataset details

1. Number of classes: 5
2. Class indices: 1, 2, 3, 4 and 5 (and 0 for the background)
3. Dataset URL: [Download from here](https://drive.google.com/file/d/1RRlTEG5JH28OJk_sHgh95DFLIi1p9_rb/view?usp=sharing)
4. Dataset size: Train data (447 images + 447 corresponding masks), Validation data (93 + 93), and Test data (90 + 90)

The intensity of each pixel in the mask image determines the class ID and instance ID of the pixel. Divide the intensity of the mask image by 4. The left digit is the class ID and the right digit is the result ID of the resulting two-digit number.

## Run in development (debugging) mode
1. Create a container called adaptis from the python base image exposing the container to one GPU and increasing the shared memory limit to 128 MB. Get into its  shell. Note that the present working directory msut contain the code and the dataset.
```
docker run --name=adaptis -v $(pwd):/app --gpus '"device=0"' --shm-size=128m -it python:3.8-slim /bin/bash
```
2. In the shell, run the following commands
```
apt-get clean
apt-get -y update
apt-get -y install python3-dev build-essential
apt-get install -y python3-opencv
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install -r app/requirements.txt
```
3. Run the file in the shell
```
python /app/train_toy_v2.py --batch-size=1 --workers=0 --gpus=0 --dataset-path=/app/<custom dataset path>
```

## Run in production mode
```
docker build --no-cache -t adaptis .
docker run --name adaptis --gpus '"device=0"' --shm-size=128m adaptis
```

## Changes implemented
1. A brute force method is implemented to extract the class IDs and instance IDs from the instance masks. The class IDs of an image are stored in a ```class_matrix``` array and the instance IDs in a ```instance_matrix``` array. The ```class_matrix``` serves as the ground truth for semantic segmentation and the processed instane mask (now divided by 4) serves as the ground truth for the AdaptIS head. 
2. As the dataset is high-dimensional (1536x1536 resolution), the batch size is limited to 1, number of point prosposals for each image of a batch is limited to 5 and number of training epochs to 10. The container is exposed to one GPU.
3. Since the number of classes is 5 now, the number of channels of the output of the segmentation head is now set to 10.
4. The minor implementations are commented in the files.

## Results 

