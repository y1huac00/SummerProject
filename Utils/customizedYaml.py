import os
import yaml

'''
A customized yaml phrasing module is created in this file.
Expected calling API:
params = 
'''


class yaml_handler:
    def __init__(self, yaml_path):
        self.path = yaml_path
        with open(yaml_path, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                self.data = data
            except yaml.YAMLError as exc:
                print(exc)
                exit(200)

    def get_data(self, keyword):
        return self.data[keyword]

    def build_new_path(self, keyword_root, keyword_tgt):
        return os.path.join(self.data[keyword_root], keyword_tgt)
