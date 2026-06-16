from pathlib import Path

from setuptools import find_packages, setup

requirements = [
    line.strip()
    for line in Path("requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#") and "git+" not in line
]

readme = Path("README.md")
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    # Distribution name on PyPI ("pip install quintessentia"); the import package
    # is still `quint`.
    name="quintessentia",
    version="1.1",
    description="Transcribe, chunk and summarize podcasts (FastAPI + Whisper + OpenAI)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Polovinkin Nikita",
    url="https://github.com/poloniki/quint",
    project_urls={
        "Source": "https://github.com/poloniki/quint",
        "Issues": "https://github.com/poloniki/quint/issues",
    },
    license="MIT",
    keywords=["whisper", "transcription", "summarization", "podcast", "fastapi", "nlp"],
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "frontend"]),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: FastAPI",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
