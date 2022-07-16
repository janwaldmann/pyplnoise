import setuptools

with open("README.md", "r") as f:
    LONG_DESC = f.read()

setuptools.setup(
    name="pyplnoise",
    version="1.4",
    description="Arbitrarily long streams of power law noise using NumPy and SciPy.",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="Jan Waldmann",
    author_email="dev@pgmail.org",
    py_modules=['pyplnoise'],
    license='BSD',
    url='https://github.com/janwaldmann/pyplnoise',
    python_requires='>=3.7,<3.99',
    install_requires=[
        "numpy >= 1.17.0",
        "scipy >= 1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 7 - Inactive",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
