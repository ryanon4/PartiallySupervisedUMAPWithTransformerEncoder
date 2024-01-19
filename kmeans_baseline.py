import pandas as pd
from tqdm import tqdm
from evaluate import evaluate_vectors_baseline
from generate import load_dataset
DATASETS = ["20newsgroups"]
for DATASET in DATASETS:
    accuracies = []
    for i in tqdm(range(0, 25)):
        data, target, vectors = load_dataset(DATASET)
        accuracy = evaluate_vectors_baseline(vectors, target)
        accuracies.append(accuracy)
    accuracy = pd.DataFrame(accuracies)
    accuracy.to_csv(DATASET + "_" + "kmeans_baseline" + ".csv")