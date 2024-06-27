Frequently Asked Questions
==========================

This page answers some of the commonly asked questions about Prom. Have a
question not answered here? File an issue on the `GitHub issue tracker
<https://github.com/anonymous/Prom/issues>`_.

.. contents:: Questions:
    :local:

What can I do with this?
------------------------

Prom is designed to enhance the robustness and performance of predictive
models during deployment, particularly for code analysis and optimization tasks.
Prom helps in identifying and correcting mispredicted samples due to data drift,
ensuring the deployed model performs reliably even with changes in application workloads and hardware.

I found a bug. How do I report it?
----------------------------------

Great! Please file an issue using the `GitHub issue tracker
<https://github.com/anonymous/Prom/issues>`_.  See
:doc:`contributing` for more details.

Do I have to use statistical assessments?
----------------------------------------

No. While Prom leverages statistical assessments to detect and manage data drift,
it is flexible and can be integrated with various machine learning techniques to
enhance model robustness during deployment.

When does Prom consider a sample “drifted”?
--------------------------------------------

Prom uses statistical assessments to determine if a sample is “drifted.”
If the credibility and confidence scores of a prediction are below a predefined threshold,
Prom flags the sample as drifted, indicating a potential misprediction due to changes in
the data distribution.
