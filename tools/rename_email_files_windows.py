# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:00:44 2019

@author: Corey Schafer source:https://www.youtube.com/watch?v=ve2pmm5JqmI
"""
def rename_files(dir_path, new_name):
    from pathlib import Path
    from os import rename
    os.chdir(Path(dir_path))
    print('Current Folder: ', os.getcwd())
    print(' ')
    print('Current files: ')
    for f in os.listdir():
        f_name, f_ext = (os.path.splitext(f))
        os.rename(f, new_name)
        print(f)
    
