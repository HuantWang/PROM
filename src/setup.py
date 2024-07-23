from distutils.core import setup
setup(
  name = 'PROM',         # How you named your package folder (MyLib)
  packages = ['prom'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A framework to enhance the robustness and performance of predictive models against changes during deployment',   # Give a short description about your library
  author = '',                   # Type in your name
  author_email = '',      # Type in your E-Mail
  url = '',   # Provide either the link to your github or to your website
  download_url = '',    # I explain this later on
  keywords = ['ML', 'CONFORMAL PREDICTION', 'DEPLOYMENT'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'mapie',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
