Getting Started
===============

Prom, an open-source Python toolkit designed to enhance the robustness and performance
of predictive models during deployment against changes such as new CPU architectures or
code patterns. Prom uses statistical assessments to identify test samples prone to mispredictions
and utilizes feedback on these samples to improve the deployed model.
Prom can be applied to various machine learning models and code analysis and optimization tasks.
It uses the `MAPIE <https://github.com/scikit-learn-contrib/MAPIE/tree/master>`_,
to expose conformal prediction.

.. contents:: Topics covered:
    :local:

Key Concepts
------------

Prom helps in addressing data drift during deployment,
ensuring the reliability of machine learning models in the face of changes,
and supports continuous improvements in the end-user environment.


.. image:: /_static/img/overview.png

The key features and capabilities of Prom include:

* **Task Definition**:
Prom provides a task definition API to define the code analysis and optimization tasks.

* **Predictive Model Robustness**:
Prom automatically identifies samples that are likely to be mispredicted due to data drift
and utilizes incremental learning to retrain the model with these samples, thus maintaining
the model's performance.

* **Deployment Time Performance**:
Prom provides APIs to load and reuse the updated models for optimizing new code.
It creates a session to apply a standard predictive model loop to optimize the input
program by using statistical assessments to select an action for a given state.

Our goal is to improve the robustness and reliability of machine learning models
during deployment, ensuring they perform as expected even with changes in application
workloads and hardware.


Installing SuperSonic
----------
Checkout to `install SuperSonic from source <https://github.com/HuantWang/SUPERSONIC>`_.
Installing from source gives you the maximum flexibility to configure the build effectively from the official source releases.
Install the latest CompilerGym release: See `INSTALL.md
<https://github.com/HuantWang/SUPERSONIC/blob/master/INSTALL.md>`_ for alternative installation methods.

If you would like to quickly try out SuperSonic or run some demo and tutorials, you can checkout install from Docker.

Checkout to `install Prom from source <https://github.com/anonymous/Prom>`_.
Installing from source gives you the maximum flexibility to configure the build effectively
from the official source releases.
Install the latest release: See `INSTALL.md <https://github.com/anonymous/Prom/blob/master/INSTALL.md>`_
for alternative installation methods.

If you would like to quickly try out Prom or run some demo and tutorials,
you can check out the installation from Docker.