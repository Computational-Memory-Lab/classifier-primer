# Classifying memory from EEG — an interactive primer

Live at **https://computational-memory-lab.github.io/classifier-primer/**

An introduction to the Computational Memory Lab's recognition-memory classifier
work, for people without a machine-learning background. Eight chapters, running
on the lab's real EEG.

## This repository is a mirror — do not edit it

The source lives in `site/` in the lab's main **Classifiers** repository
(private), and is mirrored here only because GitHub Pages will not serve a
private repo on a free plan. Anything you change here will be overwritten.

To change the site, edit `site/` in Classifiers and run `./site/publish.sh`.

Mirrored from Classifiers @ `87ff624`.

## Running it locally

```bash
python3 -m http.server 8000
```

It must be served rather than opened as a `file://` path: the pages are ES
modules and they fetch their data.
