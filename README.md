# ST-GCN-Based Karate Technique Recognition

## Spatio-Temporal Graph Convolutional Network for Karate Technique Recognition: A Comparison with LSTM

This repository contains the implementation for the AIML305 (Advanced Deep Neural Networks) course project on human movement classification using optical motion-capture data.

The project investigates whether explicitly modelling the spatial topology of the human body with a Spatio-Temporal Graph Convolutional Network (ST-GCN) improves karate technique recognition compared with a sequence-based Long Short-Term Memory (LSTM) model.

---

## Research Question

> Does explicit modelling of human-body topology with an ST-GCN improve karate technique recognition compared with sequence-only LSTM modelling under a subject-independent evaluation protocol?

---

## Motivation

Human movement is naturally structured in both space and time.

A karate technique is not only a sequence of changing coordinates; it also involves relationships between body parts such as:

- shoulders and arms
- torso and pelvis
- hips and legs
- knees and ankles
- feet and toes

A conventional sequence model such as an LSTM can model temporal dependencies, but a flattened representation does not explicitly encode the anatomical relationships between body markers.

This project therefore investigates an ST-GCN approach that represents the human body as a graph and jointly models:

1. Spatial relationships between body markers.
2. Temporal changes in those markers.

---

## Dataset

The project uses the:

**Optical Motion Capture Dataset of selected techniques in beginner and advanced Kyokushin karate athletes.**

The dataset contains optical motion-capture recordings of karate techniques performed by athletes with different experience levels.

The recordings use a Vicon optical motion-capture system with 39 reflective body markers.

The techniques considered in this project are:

- Gyaku-Zuki
- Mae-Geri
- Mawashi-Geri gedan
- Mawashi-Geri jodan
- Ushiro-Mawashi-Geri

The original dataset contains recordings under multiple execution conditions.

Dataset source:

- Scientific Data article: https://doi.org/10.1038/s41597-021-00801-5
- Figshare dataset: https://doi.org/10.6084/m9.figshare.13164848

Please refer to the original dataset publication and license information before redistribution or commercial use.

---

## Data Representation

Each processed movement sequence is represented using:

```text
T × V × C