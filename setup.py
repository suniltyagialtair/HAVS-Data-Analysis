from setuptools import setup, find_packages

setup(
    name='havs-data-analysis',
    version='1.0.0',
    description='Three-stage passive acoustic vessel detection and classification',
    author='Oravont Systems LLP',
    author_email='sunil@oravont.com',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
        'soundfile>=0.12',
        'openpyxl>=3.1',
        'matplotlib>=3.7',
    ],
    entry_points={
        'console_scripts': [
            'havs-analyse=analysis_pipeline.run_pipeline:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
