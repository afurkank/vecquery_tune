from setuptools import setup, find_packages

setup(
    name='optimize_query',
    version='0.0.5',
    description='Fine-tune BERT for vectorized query search',
    url='https://github.com/afurkank/optimize_query',
    author='Ahmet Furkan Karacık',
    author_email='furkan.karacikq@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.31.0',
        'torch>=2.0.1',
        'chromadb>=0.4.5'
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
