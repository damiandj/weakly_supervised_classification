from setuptools import setup, find_packages
import os

try:
    tag = os.environ['CI_COMMIT_TAG']
except KeyError:
    tag = "test123"

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name='weakly-supervised-classification',
    version=tag,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3'
    ],
    keywords='',
    packages=find_packages(exclude=['tests', 'assets']),
    install_requires=install_requires,
    dependency_links=[],
    extras_require={},
    package_data={},
    data_files=[],
    entry_points={
        'console_scripts': [
            'prepare_test_sets=weakly_supervised_classification.main:prepare_test_sets',
            'train_test_models=weakly_supervised_classification.main:train_test_models',
            'test_models=weakly_supervised_classification.main:test_models'
        ]
    }
)
