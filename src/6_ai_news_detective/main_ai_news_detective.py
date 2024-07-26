from utils.detection import detect

texts = ["""Due to the rapid development of AI technologies, many problems can be solved. At the same time, new challenges arise that require technologies to overcome them. This master's thesis explores the question of how a detector for AI-generated newspaper articles can be implemented using machine learning. It explains how large language models work and what characterizes the texts they generate. Various concepts and methods from the field of machine learning were presented, particularly classification algorithms and contrastive learning.

The subsequent documentation of the Python implementation of the detector shows how a large amount of AI-generated text can be generated using the OpenAI API, describes the preprocessing steps, and presents the architecture and training process of two contrastive learning models that aim to learn improved text representations to enable classification by a k-nearest neighbors algorithm.

It was found that the two-stage approach yielded promising results. In direct comparison, the setup with the supervised contrastive learning model achieved better accuracy than the one with the siamese network and also performed better in other metrics. The approach also had a good standing compared to existing detectors and showed better results than many of the common tools. Finally, it was observed that the burstiness parameter as a feature in preprocessing influenced the learning success of the contrastive learning models to such an extent that the learned vector representations became unusable."""] # insert texts to be predicted here

predictions = detect(texts)

for i, prediction in enumerate(predictions):
    if prediction == 1:
        print(f'Text no. {i} is predicted as AI.')
    else:
        print(f'Text no. {i} is predicted as Human.')