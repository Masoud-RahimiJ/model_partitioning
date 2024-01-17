method = [780, 1899, 5688, 1704, 8262, 13243, 1133, 6563, 12647]
original = [794, 1936, 7677, 1733, 5773, 18193, 1390, 9067, 17980]

"git pull origin gpu && sudo docker build -t ev_to -f evaluation/pytorch/dockerfile ."
"sudo docker run --name exp -d --rm ev_to python3 -m evaluation.pytorch.regnet && python3 memory_monitor.py"