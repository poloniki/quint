from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [
    line.strip()
    for line in content
    if line.strip() and not line.startswith("#") and "git+" not in line
]

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
