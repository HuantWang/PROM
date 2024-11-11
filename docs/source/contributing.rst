Contributing
=============

**Table of Contents**

-  `How to Contribute <#how-to-contribute>`__
-  `Pull Requests <#pull-requests>`__
-  `Leaderboard Submissions <#leaderboard-submissions>`__
-  `Code Style <#code-style>`__


--------------

How to Contribute
-----------------

We want to make contributing to Prom as easy and transparent as possible. The most helpful ways to contribute are:

1. Provide feedback.

   -  `Report bugs <https://github.com/HuantWang/PROM/issues>`__. It is particularly important to report any crash or correctness issues. We use GitHub issues to track public bugs. Ensure your description is clear and includes sufficient instructions to reproduce the issue.
   -  Report issues related to incomplete or unclear documentation, or suggest improvements for error messages.
   -  Make feature requests. Let us know if you have use cases that are not well supported, with as much detail as possible.

2. Contribute to the Prom ecosystem.

   -  Pull requests. Please see below for details. A great way to start is to pick an `unassigned “Good first issue” <https://github.com/HuantWang/PROM/issues?q=is%3Aopen+is%3Aissue+no%3Aassignee+label%3A%22Good+first+issue%22>`__!
   -  Add new features not on `the roadmap <https://your-repo-docs/roadmap>`__. Examples include implementing new conformity measures, expanding deployment-time functionality, or creating new case studies.

Pull Requests
-------------

We actively welcome your pull requests.

1. Fork `the repo <https://github.com/HuantWang/PROM>`__ and create your branch from ``main``.
2. Follow the instructions for `building from source <https://github.com/HuantWang/PROM/blob/main/INSTALL.md>`__ to set up your environment.
3. If you’ve added code that should be tested, add tests.
4. If you’ve changed APIs, update the `documentation <https://github.com/HuantWang/PROM/tree/main/docs>`__.
5. Make sure your code lints (see `Code Style <#code-style>`__ below).
6. If you haven’t already, complete the `Contributor License Agreement (“CLA”) <#contributor-license-agreement-cla>`__.

Code Style
----------

We aim to make code formatting straightforward with tools. The code style guidelines are as follows:

-  Python: Use `black <https://github.com/psf/black>`__ and `isort <https://pypi.org/project/isort/>`__.
-  C++: Follow the `Google C++ style guide <https://google.github.io/styleguide/cppguide.html>`__ with a 100-character line length and ``camelCaseFunctionNames()``.

We use `pre-commit <https://pre-commit.com/>`__ to ensure code is formatted before committing. Run pre-commit before submitting pull requests. See `INSTALL.md <https://github.com/HuantWang/PROM/blob/main/INSTALL.md>`__ for installation and usage instructions.

Other rules:

-  Use descriptive names rather than short ones.
-  Split complex code into manageable units.
-  Write deterministic tests for new features.
-  Prioritize code that is easy to use, then easy to read, and finally easy to write.



