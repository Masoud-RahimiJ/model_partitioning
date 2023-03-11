mkdir models
cd models
wget -c https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
cd ..
Use python version less than 3.8
pip install -r requirements.txt
python split_model_layers.py
Then you can run one_part or multi_part
