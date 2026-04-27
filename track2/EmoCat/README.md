# VMC2026 Track 2 Emotion Categories Baseline

Contact for questions about this baseline: ecooper@nict.go.jp

In the listening test, listeners were instructed to choose which of five
categories they perceived in the speech (Neutral, Happy, Sad, Angry, Surprise).
Listeners were allowed to choose more than one.

The task in this sub-track is to predict the proportions of votes that each
category received for each audio sample.

The baseline is category prediction probabilities from Emotion2vec+ large.

Ziyang Ma, Zhisheng Zheng, Jiaxin Ye, Jinchao Li, Zhifu Gao, ShiLiang Zhang, and
Xie Chen. 2024. emotion2vec: Self-Supervised Pre-Training for Speech Emotion
Representation. In Findings of the Association for Computational Linguistics:
ACL 2024, pages 15747–15760, Bangkok, Thailand. Association for Computational
Linguistics.  URL: https://aclanthology.org/2024.findings-acl.931/

Please follow the "Inference with checkpoints: instructions on their GitHub
README to predict emotion category probabilities for wavs in your wav directory:

   https://github.com/ddlBoJack/emotion2vec

We have provided a modified version of their example code that writes out
probabilities of our five emotion categories of interest.

   run_vmc2026.py

First, edit `indir` to point to your wav directory.
Then, run the script and find the output in `category_probs.out`.
