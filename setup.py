from setuptools import setup, find_packages

setup(
    name="bonvoyage",
    version="0.1.0",
    description="AI travel planning agent for solo travelers",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
)
