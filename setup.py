from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='VSFE',
    version=__version__,
    author='Hanying Zong',
    author_email='zonghy1225@163.com',
    url='',
    license='MIT',
    packages=find_packages(include=['cross_view_transformer', 'cross_view_transformer.*']),
    zip_safe=False,
)
