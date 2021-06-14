Repository for the Track 5.2 (Toxicity) @ DSTC10


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
- Luis F. D'Haro (Universidad Politécnica de Madrid, Spain)
- João Sedoc (New York University, USA)
- Chen Zhang (National University of Singapore, Singapore)
- Rafael Banchs (Intapp Inc., USA)
- Alexander Rudnicky (Carnegie Mellon University, USA)
- Haizhou Li (National University of Singapore, Singapore)
