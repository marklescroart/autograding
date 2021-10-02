#!/usr/bin/env python

# Start with older, fewer-featured distutils for packaging if possible
from distutils.core import setup
import sys
import os

# If more advanced features are required for install, import newer setuptools package
if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
            )).intersection(sys.argv)) > 0:
    from setuptools import setup
# For differences between distutils and setuptools, see:
# http://stackoverflow.com/questions/25337706/setuptools-vs-distutils-why-is-distutils-still-a-thing
# http://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2

if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def main(**kwargs):
    setup(name="""autograding""",
          version='0.1.0',
          description="""Code for auto-grading exams and quizzes""",
          author='Mark Lescroart',
          license='Unclear',
          url='gallantlab.org',
          packages=['autograding'],
          long_description = read('README.md'),
          **kwargs)

if __name__ == "__main__":
    main(**extra_setuptools_args)
