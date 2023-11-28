from setuptools import setup, find_packages

# Requirements lists the necessary packages and versions to run package
with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content]  # Save lines from requirements.txt

setup(
    name="facetally",  # Name of the package
    description="Set up packages for facetally project",  # Description of the package
    packages=find_packages(),  # Find packages automatically
    install_requires=requirements,
)
