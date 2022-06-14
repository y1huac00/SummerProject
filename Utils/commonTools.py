import argparse
import os
import regex as re

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./format.yaml', help='format yaml path')
    opt = parser.parse_args()
    return opt

def files(path):
  for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
      yield file

def folders(path):
  for file in os.listdir(path):
    if not os.path.isfile(os.path.join(path, file)):
      yield file

def conditional_folders(path, condition):
  '''
  return folders with name matched the condition pattern
  '''
  for file in os.listdir(path):
    if not os.path.isfile(os.path.join(path, file)) and bool(re.search(condition, file)):
      yield file