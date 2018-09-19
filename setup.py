from setuptools import setup, find_packages

requirements = [
    'numpy',
    'torch',
]

setup(
    name="pytorch_face",
    version="0.0.1",
    author="yxchng",
    description="torchvision for face recognition",
    packages=find_packages(exclude=('test', 'train')),
    install_requires=requirements,
)
