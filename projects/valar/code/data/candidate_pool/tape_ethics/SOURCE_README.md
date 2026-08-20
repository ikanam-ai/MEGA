# TAPE ethics item bank

Source: RussianNLP/tape (HuggingFace, local copy)
Subsets: per_ethics, sit_ethics
Splits: train (labeled), test (all meta labels = -1)

Label fields in meta:
  per/sit_virtue, per/sit_moral, per/sit_law, per/sit_justice, per/sit_util
  Values: 1 (positive), 0 (negative), -1 (unlabeled/test split)

Texts are 3rd-person Russian news articles about ethical situations.
Length: ~17–1016 words, median ~138 words.

Total items: 3,415
