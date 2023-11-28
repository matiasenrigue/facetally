from setuptools import setup, find_packages

# Requirements lists the necessary packages and versions to run package
# Install with !pip install -e . during development
with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content]  # Save lines from requirements.txt

setup(
    name="facetally",  # Name of the package
    version="0.0.10",
    description="Set up packages for facetally project",  # Description of the package
    packages=find_packages(),  # Find packages automatically
    install_requires=requirements,
)
