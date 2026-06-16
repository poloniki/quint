from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="quint",
    version="1.1",
    description="FastAPI service that transcribes, chunks, and summarizes podcasts",
    url="https://github.com/poloniki/quint",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=requirements,
)
