import setuptools

setuptools.setup(name='dryad',
      version='0.3a1',
      description='Prediction of Cancer Phenotypes Using Mutation Trees',
      author='Michal Radoslaw Grzadkowski',
      author_email='grzadkow@ohsu.edu',
      packages=setuptools.find_packages(
          exclude=["dryadic.tests.*", "dryadic.tests"]),
      url = 'https://github.com/ohsu-comp-bio/dryad',
      download_url = ('https://github.com/ohsu-comp-bio/'
                      'dryad/archive/v0.2.1.tar.gz'),
      install_requires=[
          'numpy>=1.14.3',
          'pandas>=0.23.3',
          'scikit-learn>=0.19.1',
          'pystan>=2.17.1',
        ]
     )

