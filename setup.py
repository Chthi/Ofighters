from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as readme_file:
    readme = readme_file.read()

name = "ofighters"

version = "0.1.0"

description = "Spaceship combat simulator made as an artificial intelligence arena. I'm playing with reinforcement learning, dense and convolutional models with multiple inputs and outputs and neural network made from scratch. "

long_description = readme


requirements = ["tensorflow", "keras", "numpy", "matplotlib", "scikit-image", "pandas"]
extras_requirements = []

setup(
    name=name,
    version=version,
    author="Thibault Charmet",
    author_email="",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chthi/Ofighters",
    project_urls={},
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    py_modules=[],
    entry_points={
        "console_scripts": [
            # package=module.file:function
            "ofighters=ofighters.__main__:main",
        ]
    },
    python_requires='>=3.6',
    install_requires=requirements,
    # extras_require=extras_requirements,
    include_package_data=True,
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords='Spaceship AI Arena',
)
