import setuptools

setuptools.setup(name='dryad',
      version='0.2',
      description='Prediction of Cancer Phenotypes Using Mutation Trees',
      author='Michal Radoslaw Grzadkowski',
      author_email='grzadkow@ohsu.edu',
      packages=setuptools.find_packages(
          include=["dryadic.features", "dryadic.learning"]),
      url = 'https://github.com/ohsu-comp-bio/dryad',
      download_url = ('https://github.com/ohsu-comp-bio/'
                      'dryad/archive/v0.1.tar.gz'),
      install_requires=[
          'numpy>=1.14,<1.15',
          'pandas>=0.23.3',
          'scikit-learn>=0.19.1',
          'pystan>=2.17.1',
        ]
     )

