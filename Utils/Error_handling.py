import pandas as pd
import commonTools
import customizedYaml

'''
The file is created for getting the files that require attention for re-tagging. An error csv file is provided for
reference. Out put of this script should be a folder holding files pending for fixing.

This script should also has the ability to:
    1. Record the fixed and un-fixed files, pack the files pending for repairing.
    2. Move the fixed files back to the dataset and replace the ones containing error
    3. update the fixed files in the error recording.

Some requirements for the design:
    a. Separate I/O and CPU work into different functions.
    b. Avoid using static variable. All static variables should be read from yaml file.
'''

if __name__=='main':
    yaml_data = customizedYaml.yaml_handler()

