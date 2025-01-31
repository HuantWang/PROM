Frequently Asked Questions
==========================

This page answers some of the commonly asked questions about Prom. Have a
question not answered here? File an issue on the `GitHub issue tracker <https://github.com/HuantWang/PROM/issues>`_.

.. contents:: Questions:
    :local:

What can I do with this?
------------------------

Prom is designed to enhance the robustness and performance of predictive
models during deployment, particularly for code analysis and optimization tasks.
Prom helps in identifying and correcting mispredicted samples due to data drift,
ensuring the deployed model performs reliably even with changes in application workloads and hardware. Prom’s ability to assess prediction credibility and confidence using conformal prediction makes it well-suited for continuous model improvement.

I found a bug. How do I report it?
----------------------------------

Great! Please file an issue using the `GitHub issue tracker <https://github.com/HuantWang/PROM/issues>`_. See the `contributing guide <https://github.com/HuantWang/PROM/blob/main/CONTRIBUTING.md>`_ for more details on how to provide useful information for debugging.

Do I have to use statistical assessments?
-----------------------------------------

No. While Prom leverages statistical assessments to detect and manage data drift,
it is flexible and can be integrated with various machine learning techniques to
enhance model robustness during deployment. This flexibility allows users to choose the most suitable methods for their specific applications.

When does Prom consider a sample “drifted”?
--------------------------------------------

Prom uses statistical assessments to determine if a sample is “drifted.”
If the credibility and confidence scores of a prediction are below a predefined threshold,
Prom flags the sample as drifted, indicating a potential misprediction due to changes in
the data distribution. Prom’s adaptive scheme for selecting calibration samples ensures accurate nonconformity score calculations, leading to better detection of drifting data patterns.

Can Prom handle both classification and regression tasks?
----------------------------------------------------------

Yes. Prom is designed to support both classification and regression models.
It uses different nonconformity functions tailored for each type of task to compute credibility and confidence scores, allowing it to handle diverse model outputs effectively.

How does Prom improve deployment-time performance?
---------------------------------------------------

Prom identifies drifting samples and suggests corrective actions through its incremental learning feature. By retraining on a small subset of flagged samples, Prom helps maintain the deployed model’s performance, reducing the need for extensive manual data labeling and frequent model retraining.

What is conformal prediction and why does Prom use it?
-------------------------------------------------------

Conformal prediction (CP) is a statistical technique that evaluates how well a prediction fits within a defined confidence interval based on training data. Prom uses CP to calculate p-values for predictions, enabling it to flag potential mispredictions by assessing both credibility and confidence scores. This method enhances the detection of data drift and supports robust decision-making during deployment.

