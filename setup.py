import os
from setuptools import setup, find_packages


def get_install_requirements():
    requirements = []
    requirements_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "requirements.txt"
    )
    with open(requirements_file) as f_req:
        for line in f_req:
            line = line.strip()
            if not line.startswith("#") and len(line) > 0:
                requirements.append(line)

    return requirements

setup(
    name="clfm",
    version="1.0.0",
    author="Bang Yang",
    author_email="yangbang@pku.edu.cn",
    description="Continual Learning of Foundation Models",
    license="Apache License 2.0",
    packages=find_packages(),
    python_requires=">=3.8.0",
    install_requires=get_install_requirements(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
