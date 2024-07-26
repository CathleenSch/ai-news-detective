# AI News Detective - A tool for detecting news articles generated by AI

```plaintext
ai_news_detective/
├── resources/                    # Contains original texts, Word2Vec model, and folders to store preprocessed and split data
├── src/
│   ├── 0_data_collection/
│   │   ├── utils/                # Sub modules
│   │   ├── collect_gpt_texts_batch.py  # Script to collect AI generated news articles via OpenAI API
│   │   ├── collect_paraphrased_texts.py  # Script to collect paraphrased versions of AI news articles via OpenAI API
│   │   ├── check_batch_job.py    # Script to check the status of a batch job running at OpenAI
│   │   ├── retrieve_batch_job_result_texts.py  # Script to retrieve results of a batch job and save them to disk
│   │   └── batch_processing_perplexity.py    # Script to create batch processes to collect token probabilities to calculate text's perplexity (beware, this can get expensive fast)
│   ├── 1_preprocessing/
│   │   ├── ai_preprocessed/ and human_preprocessed/  # Folders where preprocessed and vectorized texts are stored
│   │   ├── utils/                # Sub modules
│   │   ├── 1.1_main_preprocessing.py  # Script to execute preprocessing steps on data
│   │   └── 1.2_split_data.py    # Script to split preprocessed data into training, validation, and test sets
│   ├── 2_siamese_network/
│   │   ├── model/               # Folder in which trained models are saved
│   │   ├── utils/               # Sub modules
│   │   ├── 2.1_main_siamese_network.py  # Script to train siamese network
│   │   ├── 2.2_get_embeddings_from_siamese_model.py  # Script to generate embeddings from training and validation data to use with classifier
│   │   └── siamese_network_config.yml  # Config file to determine which version of the model should be trained
│   ├── 3_supervised_contrastive_learning/
│   │   ├── model/               # Folder in which trained models are saved
│   │   ├── utils/               # Sub modules
│   │   ├── 3.1_main_supervised_contrastive_learning.py  # Script to train supervised contrastive learning model
│   │   ├── 3.2_get_embeddings_from_supervised_cl_model.py  # Script to generate embeddings from training and validation data to use with classifier
│   │   └── supervised_config.yml  # Config file to determine which version of the model should be trained
│   ├── 4_classifier/
│   │   ├── models/              # Folder in which trained classifier models are saved
│   │   ├── siamese_network_data/  # Folder in which results from script 2.2 are stored
│   │   ├── supervised_cl_data/  # Folder in which results from script 3.2 are stored
│   │   ├── utils/               # Sub modules
│   │   ├── 4.1_main_classifier.py  # Script to carry out hyperparameter tuning on various classification algorithms
│   │   └── classifier_config.yml  # Config file to determine which embeddings from which model should be used for training
│   ├── 5_evaluation/
│   │   ├── models/              # Folder in which all previously trained models are stored
│   │   ├── utils/               # Sub modules
│   │   ├── 5.1_evaluation.py    # Script to evaluate a classification model using test data
│   │   ├── 5.2_evaluate_paraphrasing_attack.py  # Script to test classification models on paraphrased texts
│   │   └── evaluation_config.yml  # Config file to determine which model to evaluate
│   └── 6_ai_news_detective/
│       ├── models/              # Folder in which the final chosen models are stored for the application
│       ├── utils/               # Sub modules
│       └── main_ai_news_detective.py  # Main application to have a text evaluated using the final models
├── config_local_template.yml  # Config template which should be copied as config_local.yml and edited to contain the OpenAI API key
└── requirements.txt         # File that contains all necessary Python libraries to be installed before use


### Usage
* before usage, run `pip install requirements.txt`
* then you can run any script, provided you have the necessary data ready
    
