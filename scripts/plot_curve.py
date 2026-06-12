import json
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(file_path, output_path, output_path_metrics):
    train_losses = []
    val_losses = []
    rouge_scores = []
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    meteor_scores = []
    agg_scores = []

    with open(file_path, "r") as f:
        content = f.read()

    # Parser uniquement le premier JSON
    decoder = json.JSONDecoder()
    config, idx = decoder.raw_decode(content)

    # Le reste du fichier (logs)
    rest = content[idx:].strip()

    trainloss=True
    for line in rest.splitlines():
        if not line.strip():
            continue
        data = json.loads(line)

        if not trainloss:
            val_losses.append(float(data["val_CE-Loss"]))
            rouge_scores.append(float(data["val_ROUGE-L"]))
            bleu1_scores.append(float(data["val_BLEU-1"]))
            bleu2_scores.append(float(data["val_BLEU-2"]))
            bleu3_scores.append(float(data["val_BLEU-3"]))
            bleu4_scores.append(float(data["val_BLEU-4"]))
            meteor_scores.append(float(data["val_METEOR"]))
            agg_scores.append(float(data["val_agg_metrics"]))
            trainloss=True
            continue

        if trainloss:
            train_losses.append(float(data["train_loss"]))
            trainloss=False
            continue

        

    print(train_losses)
    print(val_losses)


    ypoints = np.array(train_losses)
    ypoints_val = np.array(val_losses)

    plt.plot(ypoints, marker = 'o', label="Train Loss")
    plt.plot(ypoints_val, marker = 'o', label="Validation Loss")
    plt.title("Training and Validation Loss in function of the number of epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (cross-entropy)")
    plt.legend()
    plt.savefig(output_path)

    plt.figure()
    plt.plot(rouge_scores, marker = 'o', label="ROUGE-L")
    plt.plot(meteor_scores, marker = 'o', label="METEOR")
    plt.plot(agg_scores, marker = 'o', label="Aggregate Score")
    plt.title("Evaluation Metrics in function of the number of epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(output_path_metrics.split(".png")[0] + "_other_metrics.png")

    plt.figure()
    plt.plot(bleu1_scores, marker = 'o', label="BLEU-1")
    plt.plot(bleu2_scores, marker = 'o', label="BLEU-2")
    plt.plot(bleu3_scores, marker = 'o', label="BLEU-3")
    plt.plot(bleu4_scores, marker = 'o', label="BLEU-4")
    plt.title("BLEU Metrics in function of the number of epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(output_path_metrics.split(".png")[0] + "_bleu.png")


if __name__ == "__main__":
    file_path = "/home/renault/repo/radarllm/lavis/output/radarLLM/radar_captioning/20260515100_with_adapter/log.txt"
    output_path = "/home/renault/repo/radarllm/lavis/output/radarLLM/radar_captioning/20260515100_with_adapter/train_loss_curve.png"
    output_path_metrics = "/home/renault/repo/radarllm/lavis/output/radarLLM/radar_captioning/20260515100_with_adapter/metrics_curve.png"
    plot_training_loss(file_path, output_path, output_path_metrics)

    