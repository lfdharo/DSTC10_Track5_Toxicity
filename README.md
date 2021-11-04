## Repository for the Track 5.2 (Toxicity) @ DSTC10


# DSTC10 - Track 5, Subtask 2: Moderation of Open-domain Dialogue Systems

In this task, our goal is to evaluate the capability of generative dialogue systems to generate appropriate answers that can go beyond detecting toxicity and moderate the conversation by producing appropriate and correct answers that allow the system to continue with the dialogue. For this task a dataset of pairs of 30K messages (training and validation set) with the following characteristics will be provided:

A toxic user writes a Tweet message using one or several swear words and the message is replied by another user which in turn is not using any swear word.
A toxic user posts a message in Reddit using one or several swear words and the message is replied by another user without using any swear word.
Dialogue scripts from movies will be also used, where in the first turn the user is using swearing language, while in the second turn the response does not contain any swearing language.

During the development phase, participants need to come up with systems that are capable of generating polite, specific and semantically appropriate responses in such scenarios.

During the evaluation phase, a hidden test set will be provided to the participants for them to generate system responses, which will be evaluated based on the objective similarity between the generated response and the original response (e.g. sentence embedding similarity, Deep AM-FM (Zhang et al., 2021), BLEU, ROUGE, etc). For the top-3 submitted systems in the objective evaluation, a set of 100 responses will be manually evaluated for politeness, specificity, semantically appropriateness and fluency.


# Dataset
Please register and download the data at https://chateval.org/dstc10. The dataset is composed of a subset of four different datasets:

1. MovieDic (Banchs, 2012)
2. Cornell Movie dataset (Danescu-Niculescu-Mizil and Lee, 2011)
3. ChatCorpus: the Twitter_en big dataset available at https://github.com/Marsan-Ma/chat_corpus/
4. Reddit dataset: available at https://github.com/microsoft/dstc8-reddit-corpus 

All the datasets have been passed through different pre-trained models to extract the following information:

1. Named Entities replacements: using the Stanza library (more information at https://stanfordnlp.github.io/stanza/ner.html)
2. Natural Language Inference (NLI) using Deberta (He et al., 2020) available as pre-trained model at https://huggingface.co/microsoft/deberta-large-mnli
3. Humour (Annamoradnejad and Zoghi, 2020): pre-trained model available at https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection  
4. Sarcasm: pre-trained model available at https://huggingface.co/mrm8488/t5-base-finetuned-sarcasm-twitter 
5. Emotions (Rodriguez-Cantelar and D'Haro, 2021): 7 different emotions (happiness, sadness, fear, angry, surprise, disgust, neutral) trained on Carer (Saravia et al., 2018), DailyDialog (Li et al., 2017), EmpathicDialogs (Rashkin et al., 2018), and EmotionLines (Chen et al., 2018). This is a DistilBertModel + Multiclass classifier (Softmax), with 6 Hidden layers, trained during 10 epochs, a vocabulary of 30K, embedding size of 768 and max. length of 512. Trained using Flair library.
6. Ratios: Length of the toxic turn vs answer turn
7. Toxicity from Perspective API: for the toxic and answer turns. API available at https://developers.perspectiveapi.com/s/about-the-api8. 


# Statistics training dataset

| Dataset                    | MovieDIC          |    Cornell      |      Twitter          |     Reddit          |
|----------------------------|-------------------|-----------------|-----------------------|---------------------|
| No. Turns                  | 3359              |    1829         | 74093                 |  32977              |
| Avg. turn length toxic     | 16.3              |    17.3         | 15.5                  |  24.1               |
| Avg. turn length answer    | 9.3               |    9.5          | 11.7                  |  15.8               |
| Avg. humour toxic          | 0.95              |    0.95         | 0.92                  |  0.95               |
| Avg. humour answer         | 0.78              |    0.79         | 0.81                  |  0.85               |
| Avg. sarcasm toxic         | 0.53              |    0.53         | 0.61                  |  0.62               |
| Avg. sarcasm answer        | 0.45              |    0.44         | 0.51                  |  0.54               |
| Avg. contradiction (NLI)   | 0.41              |    0.41         | 0.31                  |  0.26               |
| Avg. neutral (NLI)         | 0.55              |    0.55         | 0.66                  |  0.72               |
| Avg. entailment (NLI)      | 0.03              |    0.04         | 0.03                  |  0.03               |
| Major emotion toxic        | Anger (38.7%)     | Anger (37.8%)   | Anger (35.2%)         |  Anger (33.7%)      |
| Major emotion answer       | Neutral (62.4%)   | Neutral (61.2%) | Happiness (29.3%)     |  Neutral (28.4%)    |
| Avg. Perspective toxic     | 0.79              |    0.77         | 0.81                  |  0.80               |
| Avg. Perspective answer    | 0.15              |    0.15         | 0.22                  |  0.16               |


# Statistics dev dataset

| Dataset                    | MovieDIC          |    Cornell      |      Twitter          |     Reddit          |
|----------------------------|-------------------|-----------------|-----------------------|---------------------|
| No. Turns                  | 720               |    392          | 15877                 |  7066               |
| Avg. turn length toxic     | 16.3              |    17.5         | 15.5                  |  24.1               |
| Avg. turn length answer    | 9.5               |    9.4          | 11.8                  |  15.8               |
| Avg. humour toxic          | 0.94              |    0.95         | 0.92                  |  0.95               |
| Avg. humour answer         | 0.78              |    0.80         | 0.81                  |  0.85               |
| Avg. sarcasm toxic         | 0.55              |    0.53         | 0.62                  |  0.63               |
| Avg. sarcasm answer        | 0.45              |    0.45         | 0.51                  |  0.53               |
| Avg. contradiction (NLI)   | 0.44              |    0.42         | 0.31                  |  0.26               |
| Avg. neutral (NLI)         | 0.51              |    0.53         | 0.66                  |  0.71               |
| Avg. entailment (NLI)      | 0.04              |    0.05         | 0.03                  |  0.03               |
| Major emotion toxic        | Anger (40.8%)     | Anger (37.0%)   | Anger (35.9%)         |  Anger (33.9%)      |
| Major emotion answer       | Neutral (61.5%)   | Neutral (65.0%) | Happiness (29.4%)     |  Neutral (14.9%)    |
| Avg. Perspective toxic     | 0.80              |    0.77         | 0.80                  |  0.80               |
| Avg. Perspective answer    | 0.16              |    0.14         | 0.22                  |  0.16               |


# Statistics test dataset

| Dataset                    | MovieDIC          |    Cornell      |      Twitter          |     Reddit          |
|----------------------------|-------------------|-----------------|-----------------------|---------------------|
| No. Turns                  | 1822              |  995            | 15879                 | 7067                |
| Avg. turn length toxic     | 20.6              |  15.3           | 15.5                  | 24.0                |
| Avg. turn length answer    | 9.1               |  8.7            | 11.6                  | 15.7                |
| Avg. humour toxic          | 0.96              |  0.92           | 0.92                  | 0.95                |
| Avg. humour answer         | 0.78              |  0.77           | 0.80                  | 0.84                |
| Avg. sarcasm toxic         | 0.51              |  0.53           | 0.61                  | 0.61                |
| Avg. sarcasm answer        | 0.42              |  0.44           | 0.51                  | 0.53                |
| Avg. contradiction (NLI)   | 0.41              |  0.41           | 0.32                  | 0.26                |
| Avg. neutral (NLI)         | 0.55              |  0.54           | 0.66                  | 0.72                |
| Avg. entailment (NLI)      | 0.05              |  0.05           | 0.02                  | 0.03                |
| Major emotion toxic        | Anger (36.3%)     |  Neutral (39.1%)| Anger (35.7%)         | Anger (34.5%)       |
| Major emotion answer       | Neutral (57.1%)   |  Neutral (63.8%)| Happiness (29.7%)     | Neutral (28.3%)     |
| Avg. Perspective toxic     | 0.79              |  0.65           | 0.81                  | 0.80                |
| Avg. Perspective answer    | 0.15              |  0.14           | 0.22                  | 0.16                |



# Submission

Participants are expected to take the NRE columns (both the toxic and answer) and train systems that learn to generate polite and appropriated answers to the toxic comments. 
The NRE columns will be the ones used for evaluating the submitted systems.
We ask participants to use the NRE columns since it contain tokenized text using Spacy, as well as recognized named-entities replaced with labels to keep the portability and robustness of the proposed systems. 

Participants may use the different columns to filter the provided data. Besides, participants can add new features to allow them for a better selection of the answers that their systems will be using for learning good answers.

Submitted systems will be requested to provide the ID of the toxic comment and the generated answer which must be tokenized and with the replaced name-entities.

Please subscribe to this repository to keep track of changes in the data, metrics, baselines, etc.

# Citation

If you use this data or the baseline model, please cite our [paper](http://arxiv.org/abs/2111.02110):

> @article{zhang2021auteval,
>      title={"Automatic Evaluation and Moderation of Open-domain Dialogue Systems},
>      author={Zhang Chen and João Sedoc and Luis Fernando D’Haro and Rafael Banchs and Alexander Rudnicky},
>      year={2021},
>      eprint={2111.02110},
>      archivePrefix={arXiv},
>      primaryClass={cs.CL}
>}



# References:
- Annamoradnejad, I., & Zoghi, G. (2020). Colbert: Using bert sentence embedding for humor detection. arXiv preprint arXiv:2004.12765.
- Banchs, R. E. (2012, July). Movie-DiC: a movie dialogue corpus for research and development. In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 203-207).
- Chen, S. Y., Hsu, C. C., Kuo, C. C., & Ku, L. W. (2018). Emotionlines: An emotion corpus of multi-party conversations. arXiv preprint arXiv:1802.08379.
- Danescu-Niculescu-Mizil, C., & Lee, L. (2011). Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs. arXiv preprint arXiv:1106.3077.
- He, P., Liu, X., Gao, J., & Chen, W. (2020). Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654.
- Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). Dailydialog: A manually labelled multi-turn dialogue dataset. arXiv preprint arXiv:1710.03957.
- Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2018). Towards empathetic open-domain conversation models: A new benchmark and dataset. arXiv preprint arXiv:1811.00207.
- Rodriguez-Cantelar, M., D'Haro, L. F. (2021). Pretrained model available at https://drive.google.com/file/d/1lbEHWOFQt66n-T06cLYDbEhSmXlju4vx/view?usp=sharing
- Saravia, E., Liu, H. C. T., Huang, Y. H., Wu, J., & Chen, Y. S. (2018). Carer: Contextualized affect representations for emotion recognition. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 3687-3697).


# Organizers
- Luis Fernando D'Haro (Universidad Politécnica de Madrid, Spain)
- João Sedoc (New York University, USA)
- Chen Zhang (National University of Singapore, Singapore)
- Rafael Banchs (Intapp Inc., USA)
- Alexander Rudnicky (Carnegie Mellon University, USA)
- Haizhou Li (National University of Singapore, Singapore)
