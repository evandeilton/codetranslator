from setuptools import setup, find_packages

setup(
    name='codetranslator',
    version='0.1.1',
    author='Jose Lopes',
    author_email='evandeilton@gmail.com',
    description='A code translation and analysis tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/evandeilton/codetranslator',  # Replace with actual repository URL if available
    packages=find_packages(),
    install_requires=[
        'openai',
        'anthropic',
        'python-dotenv',
        'rich',
        'tiktoken'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'codetranslator=codetranslator:main',  # Adjust this based on your main entry point
        ],
    },
)