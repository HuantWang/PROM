Getting Started
===============

Prom, an open-source Python toolkit, is designed to enhance the robustness and performance
of predictive models during deployment against changes such as new CPU architectures or
code patterns. Prom uses statistical assessments to identify test samples prone to mispredictions
and utilizes feedback on these samples to improve the deployed model.
Prom can be applied to various machine learning models and code analysis and optimization tasks.
It uses `MAPIE <https://github.com/scikit-learn-contrib/MAPIE/tree/master>`_ to expose conformal prediction.

.. contents:: Topics covered:
    :local:

Key Concepts
------------

Prom helps address data drift during deployment,
ensuring the reliability of machine learning models in the face of changes,
and supports continuous improvements in the end-user environment.

.. image:: /img/workflow_simple.pdf

The key features and capabilities of Prom include:

* **Task Definition**:
Prom provides a task definition API to define code analysis and optimization tasks.

* **Predictive Model Robustness**:
Prom automatically identifies samples that are likely to be mispredicted due to data drift
and utilizes incremental learning to retrain the model with these samples, thus maintaining
model performance. It achieves this by computing credibility and confidence scores based on conformal prediction (CP) theory to assess the reliability of each prediction.

* **Deployment Time Performance**:
Prom provides APIs to load and reuse updated models for optimizing new code.
It creates a session to apply a standard predictive model loop to optimize the input
program by using statistical assessments to select an action for a given state.

Prom leverages an adaptive approach to select subsets of calibration samples and applies weighting strategies to compute nonconformity scores, making it effective in handling diverse data patterns. This improves the accuracy of detecting mispredictions and supports continuous learning with minimal user intervention by identifying and retraining on only the most critical data samples.

Our goal is to improve the robustness and reliability of machine learning models
during deployment, ensuring they perform as expected even with changes in application
workloads and hardware. Prom has demonstrated its ability to identify up to 97% of mispredictions and achieve substantial performance improvements with incremental updates.

Installing Prom
---------------
If you would like to quickly try out Prom or run some demos and tutorials,
you can check out installation options from Docker or Jupyter notebooks on online servers.

Refer to `install Prom from source <https://github.com/HuantWang/PROM>`_.
Installing from source gives you maximum flexibility to configure the build effectively from the official source releases.

