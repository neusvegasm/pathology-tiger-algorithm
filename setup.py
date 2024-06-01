from setuptools import setup, find_packages

setup(
    name="tiger_nnunet_v2",
    version="2",
    author="Simon Reichert",
    author_email="simon.reichert@ru.nl",
    packages=find_packages(),
    license="LICENSE.txt",
    install_requires=[
        "wheel==0.43.0",
        "numpy==1.26.4",
        "tqdm==4.66.4",
        "nnunetv2==2.4.2",
        "scikit-image==0.23.2"
    ],
)