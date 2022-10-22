from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setup(
    name="FaceMasque",
    version="1.0",
    description="A Python package to classify weather a person is weared a mask or not.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Kushal0189/Detection-of-Person-With-or-Without-Mask",
    author="KUSHAL MASTER",
    author_email="kushalmaster8@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["Detection-of-Person-With-or-Without-Mask-master"],
    include_package_data=True,
    install_requires=[
        "onnx==1.6.0",
        "onnx-tf==1.3.0",
        "onnxruntime==0.5.0",
        "opencv-python==4.1.1.26",
        "tensorflow==1.15.2",
        "keras",
    ],
)
