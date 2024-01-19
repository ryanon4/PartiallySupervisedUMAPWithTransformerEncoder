import pandas as pd
from generate import generate_vectors, load_dataset
from evaluate import evaluate_vectors
from tqdm import tqdm
import datetime
import time
import joblib
DATASETS = ["20newsgroups", "trec6", "trec50", "agnews"]
#["LDA", "PCA", "TSNE","UMAP", "PARAMETRIC_UMAP", "UMAP_SUPERVISED", "PARAMETRIC_UMAP_SUPERVISED","UMAP_SUPERVISED_RNN", "UMAP_SUPERVISED_TRANSFORMER"]
# For Each Algorithm
for DATASET in DATASETS:
    for algorithm in ["PARAMETRIC_UMAP", "PARAMETRIC_UMAP_SUPERVISED"]:
        accuracies = []
        runtimes = []
        print("Algorithm: " + algorithm)
        target, vectors = load_dataset(DATASET)
        for i in tqdm(range(0, 25)):
            reduced_vectors = []
            times = []
            for dimension in range(1, 17, 1):
                start_time = time.perf_counter()
                reduced_vector = generate_vectors(algorithm, target, vectors, dimension, test_size=0.8, batch_size=2048)
                finish_time = time.perf_counter()
                elapsed_time = finish_time - start_time
                times.append(elapsed_time)
                #joblib.dump(reduced_vectors, DATASET+"_"+algorithm+"_reduced_vectors"+str(i).pkl)
                reduced_vectors.append(reduced_vector)
                del reduced_vector
            accuracy = evaluate_vectors(reduced_vectors, target)
            del reduced_vectors
            del target
            runtimes.append(times)
            accuracies.append(accuracy)
        accuracies = pd.DataFrame(accuracies)
        accuracies.to_csv(DATASET+"_"+algorithm+"_"+str(datetime.datetime.now().strftime("%Y-%m-%d"))+".csv")
        runtimes = pd.DataFrame(runtimes)
        runtimes.to_csv(DATASET+"_"+algorithm+"_"+str(datetime.datetime.now().strftime("%Y-%m-%d"))+"_runtimes.csv")



