import os

# Useful functions used across training and testing scripts

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

