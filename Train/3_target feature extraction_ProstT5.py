import torch
import pandas as pd
from transformers import AutoTokenizer, T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import torch.nn as nn
import time


def load_model(device, model_path):
    print(f"[{time.ctime()}] Start loading Tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    print(f"[{time.ctime()}] Tokenizer loaded successfully")

    print(f"[{time.ctime()}] Start loading T5EncoderModel")
    model = T5EncoderModel.from_pretrained(model_path).to(device)
    print(f"[{time.ctime()}] T5EncoderModel loaded successfully")

    model.eval()  # Set to evaluation mode
    return model, tokenizer


def add_space_to_sequence(sequence):
    return " ".join(sequence)


def parse_fasta(file_path):
    print(f"[{time.ctime()}] Start parsing FASTA file: {file_path}")
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    protein_name = ""
    sequence = ""
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if protein_name:
                data.append((protein_name, add_space_to_sequence(sequence)))
            protein_name = line[1:]
            sequence = ""
        else:
            sequence += line
    if sequence:
        data.append((protein_name, add_space_to_sequence(sequence)))
    print(f"[{time.ctime()}] FASTA file parsing completed, total {len(data)} sequences")
    return data


def extract_features(fasta_file, model, tokenizer, device, batch_size=8):
    print(f"[{time.ctime()}] Start extracting features")
    sequences = [add_space_to_sequence(seq[1]) for seq in parse_fasta(fasta_file)]
    print(f"[{time.ctime()}] Sequence reading completed, starting Tokenizer encoding")

    inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    print(f"[{time.ctime()}] Tokenizer encoding completed, starting DataLoader preparation")
    dataset = TensorDataset(input_ids, attention_mask)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    all_embeddings = []
    print(f"[{time.ctime()}] Start batch inference")
    for batch in tqdm(data_loader, desc="Processing batches", unit="batch"):
        batch_input_ids, batch_attention_mask = batch
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

        embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"[{time.ctime()}] Feature extraction completed")
    return all_embeddings


def save_features_as_csv(embeddings, output_file):
    print(f"[{time.ctime()}] Start saving features to CSV file: {output_file}")
    embeddings_np = embeddings.cpu().numpy()

    # Name each feature column
    column_names = [f'feature_{i}' for i in range(embeddings_np.shape[1])]

    # Create DataFrame
    df = pd.DataFrame(embeddings_np, columns=column_names)

    # Save as CSV
    df.to_csv(output_file, index=False)
    print(f"[{time.ctime()}] Features saved to {output_file}")


def main():
    print(f"[{time.ctime()}] Program started running")
    fasta_file = "./final_target_fasta/negative_test_target.txt"
    output_file = "negative_test_target.csv"
    model_path = "./ProstEDTI/Predictor/models/ProstT5"

    print("Files in model path:", os.listdir(model_path))
    if os.path.exists(model_path):
        print("File exists and is accessible.")
    else:
        print("File is missing or inaccessible.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model(device, model_path)

    embeddings = extract_features(fasta_file, model, tokenizer, device, batch_size=4)

    save_features_as_csv(embeddings, output_file)
    print(f"[{time.ctime()}] Program execution completed")


if __name__ == "__main__":
    main()