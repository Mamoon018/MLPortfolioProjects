from setuptools import find_packages, setup 
from typing import List 
## This setup.py file is used to create an Application with your code, 
## that can be used as a package by anyone. So, here we will write the details of the package in the form of parameters.
## Example: we can import our application in the same way as we import pandas, seaborm etc.

HYPEN_E_DOT='-e .'

def get_requirements (file_path:str)->List[str]:
    
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
name= 'MLProject',
version= '0.0.1',
author= 'Mamoon',
author_email= 'mamonhaider11@gmail.com',
packages= find_packages(),
install_requires =get_requirements('requirements.txt')
## It will include all the libraries given in the requirement.txt as requirement for package.
)