sudo docker build -t alexnet_one_part -f ./alexnet/dockerfile_one .
sudo docker build -t alexnet_multi_part -f ./alexnet/dockerfile_multi .
sudo docker build -t resnet_one_part -f ./resnet101/dockerfile_one .
sudo docker build -t resnet_multi_part -f ./resnet101/dockerfile_multi .