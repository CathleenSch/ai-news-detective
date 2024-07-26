import utils.preprocessing as pp

pp.preprocess_data(
    '../../resources/NeuralNews/real_arts',
    '../../resources/gpt-3.5', 
    '../../resources/paraphrased',
    './human_preprocessed/', 
    './ai_preprocessed/'
    )