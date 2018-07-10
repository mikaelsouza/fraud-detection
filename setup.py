from setuptools import setup

setup(name='fraudutils',
      version='0.0.1',
      description='A simple lib with util functions',
      url='',
      author='Mikael Souza',
      author_email='mss2@icomp.ufam.edu.br',
      license='MIT',
      packages=['.'],
      zip_safe=False,
      install_requires=[
         'matplotlib',
         'numpy',
      ]
      )
