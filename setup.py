import setuptools

setuptools.setup(name='dryad',
      version='0.4.1',
      description='Prediction of Cancer Phenotypes Using Mutation Trees',
      author='Michal Radoslaw Grzadkowski',
      author_email='grzadkow@ohsu.edu',
      packages=setuptools.find_packages(
          exclude=["dryadic.tests.*", "dryadic.tests"]),
      url = 'https://github.com/ohsu-comp-bio/dryad',
      download_url = ('https://github.com/ohsu-comp-bio/'
                      'dryad/archive/v0.4.tar.gz'),
      install_requires=[
          'numpy>=1.16',
          'pandas>=0.25',
          'scikit-learn>=0.21',
          'pystan>=2.19',
        ]
     )

