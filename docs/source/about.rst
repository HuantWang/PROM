About
=====

Prom is a library designed to enhance the robustness and performance of predictive models
during deployment, particularly for code analysis and optimization tasks. Prom supports customizable statistical assessment techniques to target a wide range of optimization challenges. A key feature of Prom is its ability to identify and correct mispredicted samples due to data drift, ensuring the deployed model remains reliable even with changes in application workloads and hardware.

.. contents:: Overview:
    :local:

Motivation
----------

Efforts have been made to provide robust machine learning models for code analysis and optimization tasks. While these solutions show promising results, they can be brittle, as minor changes in hardware or application workloads can compromise decision accuracy, affecting model robustness. The primary issue is “data drift,” a mismatch between training and test data distributions, which undermines the performance of ML-based solutions during deployment.

As application workload patterns or low-level compiler libraries and hardware design evolve, developers must ensure their ML models remain accurate. This process of maintaining model reliability despite changes can be challenging and requires a system that can adapt to these changes in real-time.

Our Vision
----------

We present Prom to address the challenges of data drift during deployment and support continuous improvements in ML models for code analysis and optimization. Our main contributions include:

1. Introducing a generic framework to automatically identify and correct mispredicted samples during deployment.
2. Demonstrating how statistical assessments can enhance the robustness of predictive models in various code optimization tasks.
3. Providing a comprehensive evaluation validating the effectiveness of Prom in detecting and managing data drift across multiple machine learning models and code optimization scenarios.

