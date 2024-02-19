
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    dependencies = f.read().splitlines()
    
setup(
    name="HKKLLM",
    version='0.0.1-alpha',
    author='William Stigall, Hailey Walker',
    author_email="williamastigall@gmail.com, hnwalker@gmail.com",
    description=(
        "Collection of packages for LLM development, finetuning, and testing. "
        "HKLLM consists of three subpackages: "
        "promptlm: collection of functions for Generative Models. "
        "torchlm: collection of functions for Discriminative Models built in PyTorch. "
        "tflm: collection of functions for Discriminative Models built in Tensorflow."
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/CS-Senior-Project-HK-02/HKLLM",
    packages=find_packages(),
    install_requires=dependencies,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6'
)
