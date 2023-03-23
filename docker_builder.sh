sudo docker build -t alexnet_one_part -f ./alexnet/dockerfile_one .
sudo docker build -t alexnet_multi_part -f ./alexnet/dockerfile_multi .
sudo docker build -t resnet_one_part -f ./resnet101/dockerfile_one .
sudo docker build -t resnet_multi_part -f ./resnet101/dockerfile_multi .
sudo docker build -t vgg19_one_part -f ./vgg19/dockerfile_one .
sudo docker build -t vgg19_multi_part -f ./vgg19/dockerfile_multi .
sudo docker build -t regnet_one_part -f ./regnet/dockerfile_one .
sudo docker build -t regnet_multi_part -f ./regnet/dockerfile_multi .