from setuptools import setup

setup(
    name='lpr',
    version='1.0.0',
    install_requires=[
        'networkx~=2.6.3',
        'pip==22.0.4',
        'wheel==0.37.1',
        'numpy==1.22.3',
        'setuptools==60.9.1',
        'pytz~=2021.3',
        'matplotlib~=3.5.1',
        'Pillow~=9.0.1',
    ],
    packages=[
        'LPR',
        'LPR.data',
        'LPR.algorithms',
        'LPR.utils',
    ],
)
