import setuptools
from setuptools import setup
def requirements_list():
    list_of_req = []
    with open('requirements.txt') as req:
        for line in req:
            list_of_req.append(line)

    return list_of_req

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="insiders",
    version="0.0.1",
    author="EmpyreanAI",
    author_email="author@example.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EmpyreanAI/Insiders",
    packages=setuptools.find_packages(),
    install_requires=requirements_list(),
    include_package_data=True,
    package_data={"":['saved_models/*.h5']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)