from distutils.core import setup

setup(name='dryad',
      version='0.1',
      description='Prediction Cancer Phenotypes Using Mutation Dryads',
      author='Michal Radoslaw Grzadkowski',
      author_email='grzadkow@ohsu.edu',
      package_dir={'dryad': ''},
      packages=['dryad'],
      url = 'https://github.com/ohsu-comp-bio/dryad',
      download_url = ('https://github.com/ohsu-comp-bio/'
                      'dryad/archive/v0.1.tar.gz'),
     )

