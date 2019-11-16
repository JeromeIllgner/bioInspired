from setuptools import setup

setup(
    name='network',
    version='0.1',
    py_modules=['network'],
    install_requires=['Click'],
    entry_points='''
        [console_scripts]
        network=__main__:network
    ''',
)
