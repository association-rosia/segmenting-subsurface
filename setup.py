from setuptools import setup, find_packages

setup(
    name='segmenting-subsurface',
    version='1.0',
    author='RosIA',
    author_email='rosialab31@gmail.com',
    packages=find_packages(),
    install_requires=[
        'jupyter==1.0.0',
        'PyYAML==6.0.1',
        'tqdm==4.66.1',
        'numpy==1.26.3',
        'pandas==2.1.4',
        'opencv-python==4.9.0.80',
        'scikit-learn==1.3.2',
        'torch==2.1.2',
        'torchvision==0.16.2',
        'pytorch-lightning==2.1.3',
        'transformers==4.37.1',
        'wandb==0.16.2',
        'matplotlib==3.8.2',
        'plotly==5.18.0',
    ],
)
