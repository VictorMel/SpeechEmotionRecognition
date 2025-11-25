"""Audio feature extraction module for Emotion Detection.

Design Goals:
- Centralize all audio loading & feature engineering in one place.
- Easy extension: add new feature builders by registering functions.
- Caching support: avoid recomputation for large datasets.
- Minimal external dependencies: prefer torchaudio > librosa when possible.

Future Extensions:
- GPU accelerated feature extraction (torch.compile / batched processing).
- Streaming windowed extraction for real-time inference.
- On-device (edge) reduced precision support (int8 / float16 path).
"""

from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

def PlotLabelsHistogram(vY: np.ndarray, labels_list, hA: Optional[plt.Axes] = None ) -> plt.Axes:
    if hA is None:
        hF, hA = plt.subplots(figsize = (8, 6))
    vLabels, vCounts = np.unique(vY, return_counts = True)
    hA.bar(vLabels, vCounts, width = 0.9, align = 'center')
    addlabels(vLabels, vCounts)
    hA.set_title('Histogram of Classes / Labels')
    hA.set_xlabel('Class')
    hA.set_xticks(vLabels, labels_list)
    hA.set_ylabel('Count')
    return hA

def PlotSplitedDataHistogram(train_labels, test_labels, labels_list):
    plt.figure(figsize=(16, 4))
    ax = plt.subplot(2,1,1)
    PlotLabelsHistogram(train_labels,labels_list,ax)
    ax = plt.subplot(2,1,2)
    PlotLabelsHistogram(test_labels,labels_list,ax)
    plt.show()
