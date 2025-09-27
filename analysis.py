"""
Common analysis functions for embedding extraction and evaluation.
Extracted from notebooks to provide reusable analysis utilities.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

def load_encoder_weights(checkpoint_path, model):
    """Load encoder weights from a full UNet checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Extract the state_dict from PyTorch Lightning checkpoint if needed
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("Loading from PyTorch Lightning checkpoint")
        state_dict = checkpoint["state_dict"]
    else:
        print("Loading from standard model checkpoint")
        state_dict = checkpoint

    # Extract encoder weights from full model checkpoint
    print("Extracting encoder weights from full model checkpoint")
    encoder_state_dict = {}
    for key, value in state_dict.items():
        # Look for encoder weights in the checkpoint
        if key.startswith("model.encoder."):
            # Remove "model.encoder." prefix
            new_key = key.replace("model.encoder.", "")
            encoder_state_dict[new_key] = value
        elif key.startswith("encoder."):
            # Remove "encoder." prefix
            new_key = key.replace("encoder.", "")
            encoder_state_dict[new_key] = value

    print(f"Extracted {len(encoder_state_dict)} encoder weights")
    model.load_state_dict(encoder_state_dict)
    print("Encoder weights loaded successfully!")


def extract_embeddings(model, device, data_loader, use_bottleneck=True):
    """Extract embeddings from a dataset using the PDDataset format."""
    model.eval()
    embeddings = []
    metadata = []

    with torch.no_grad():
        for i in tqdm(range(len(data_loader.dataset)), desc=f"Extracting embeddings"):

            # Load data from PDDataset (returns dictionary with multiple keys)
            sample = data_loader.dataset[i]
            image = sample["image"].unsqueeze(0).to(device)  # Add batch dimension

            # Get embedding from the model
            feature_maps = model(image)
            
            if use_bottleneck and isinstance(feature_maps, list):
                # For UNet encoder, use the bottleneck (last feature map)
                embedding = feature_maps[-1].flatten()
            elif isinstance(feature_maps, list):
                # Use all feature maps concatenated
                embedding = torch.cat([fm.flatten() for fm in feature_maps])
            else:
                # For single tensor output (e.g., DINOv2)
                embedding = feature_maps.flatten()

            embeddings.append(embedding.detach().cpu().numpy())

            # Store metadata from the PDDataset format
            metadata.append(
                {
                    "disease_status": sample["disease_status"].item(),  # 1=PD, 0=HC
                    "sex": sample["sex"].item(),  # 1=M, 0=F
                    "study": sample["study"].item(),  # Study ID
                    "scanner_type": sample["scanner_type"].item(),  # Scanner type ID
                }
            )

    embeddings = np.array(embeddings)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    print(f"Number of samples: {len(metadata)}")

    return embeddings, metadata


def evaluate_knn_classification(X_train, y_train, X_test, y_test, task_name, k=5):
    """Evaluate K-NN classification performance."""
    print(f"\n=== {task_name.upper()} CLASSIFICATION ===")

    # Train K-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print(f"K-NN (k={k}) Accuracy: {accuracy:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(3, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(np.unique(y_test)),
        yticklabels=sorted(np.unique(y_test)),
    )
    plt.title(f"Confusion Matrix - {task_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return accuracy, y_pred


def knn_results(train_val_embeddings, train_val_disease, test_embeddings, test_disease, 
                train_val_sex, test_sex, train_val_study, test_study, 
                train_val_scanner, test_scanner, task_name, k=5):
    """Run K-NN evaluation on all classification tasks."""
    results = {}

    # Disease Status Classification (PD vs HC)
    disease_accuracy, disease_predictions = evaluate_knn_classification(
        train_val_embeddings, train_val_disease, test_embeddings, test_disease, 
        "Disease Status", k=k
    )
    results["Disease Status"] = disease_accuracy

    # Sex Classification (Male vs Female)
    sex_accuracy, sex_predictions = evaluate_knn_classification(
        train_val_embeddings, train_val_sex, test_embeddings, test_sex, "Sex", k=k
    )
    results["Sex"] = sex_accuracy

    # Study Classification
    study_accuracy, study_predictions = evaluate_knn_classification(
        train_val_embeddings, train_val_study, test_embeddings, test_study, "Study", k=k
    )
    results["Study"] = study_accuracy

    # Scanner Type Classification
    scanner_accuracy, scanner_predictions = evaluate_knn_classification(
        train_val_embeddings, train_val_scanner, test_embeddings, test_scanner, 
        "Scanner Type", k=k
    )
    results["Scanner Type"] = scanner_accuracy

    return results


class LinearProbe(nn.Module):
    """Linear probe for classification tasks."""
    
    def __init__(self, input_dim, num_classes, dropout_rate=0.1):
        super(LinearProbe, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def train_linear_probe(
    X_train, y_train, X_val, y_val, task_name, num_classes,
    num_epochs=100, batch_size=32, learning_rate=0.001, patience=10
):
    """
    Train a linear probe with early stopping based on validation performance.
    """
    print(f"\n{'=' * 60}")
    print(f"TRAINING LINEAR PROBE: {task_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of classes: {num_classes}")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = LinearProbe(X_train.shape[1], num_classes)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_acc = correct / total
        val_accuracies.append(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final validation evaluation
    model.eval()
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            val_predictions.extend(predicted.numpy())
            val_targets.extend(batch_y.numpy())

    val_accuracy = accuracy_score(val_targets, val_predictions)
    print(f"\nFinal Validation Accuracy: {val_accuracy:.4f}")
    print(f"Classification Report:")
    print(classification_report(val_targets, val_predictions))

    return model, val_accuracy, train_losses, val_accuracies, val_predictions, val_targets


def evaluate_model_on_test_set(model, X_test, y_test, task_name):
    """Evaluate trained model on test set."""
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        predictions = predicted.numpy()

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{task_name} - Test Set Evaluation:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return accuracy, predictions


def plot_tsne(embeddings, labels, title="t-SNE Visualization", perplexity=30, n_components=2):
    """Plot t-SNE visualization of embeddings."""
    print(f"Computing t-SNE for {len(embeddings)} samples...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
    
    return embeddings_2d


def extract_labels_from_metadata(metadata, label_type):
    """Extract labels from metadata for a specific task."""
    if label_type == "disease_status":
        return np.array([m["disease_status"] for m in metadata])
    elif label_type == "sex":
        return np.array([m["sex"] for m in metadata])
    elif label_type == "study":
        return np.array([m["study"] for m in metadata])
    elif label_type == "scanner_type":
        return np.array([m["scanner_type"] for m in metadata])
    else:
        raise ValueError(f"Unknown label type: {label_type}")


def knn_evaluation(embeddings_dict, metadata_dict, model_name, k=5):
    """Run K-NN evaluation on all classification tasks."""
    print(f"\n{'=' * 60}")
    print(f"K-NN EVALUATION: {model_name.upper()}")
    print(f"{'=' * 60}")
    
    # Extract data
    train_embeddings = embeddings_dict["train"]
    val_embeddings = embeddings_dict["val"]
    test_embeddings = embeddings_dict["test"]
    
    train_metadata = metadata_dict["train"]
    val_metadata = metadata_dict["val"]
    test_metadata = metadata_dict["test"]
    
    # Combine train and val for K-NN (as done in notebooks)
    train_val_embeddings = np.vstack([train_embeddings, val_embeddings])
    train_val_metadata = train_metadata + val_metadata
    
    # Extract labels for all tasks
    train_val_disease = extract_labels_from_metadata(train_val_metadata, "disease_status")
    test_disease = extract_labels_from_metadata(test_metadata, "disease_status")
    
    train_val_sex = extract_labels_from_metadata(train_val_metadata, "sex")
    test_sex = extract_labels_from_metadata(test_metadata, "sex")
    
    train_val_study = extract_labels_from_metadata(train_val_metadata, "study")
    test_study = extract_labels_from_metadata(test_metadata, "study")
    
    train_val_scanner = extract_labels_from_metadata(train_val_metadata, "scanner_type")
    test_scanner = extract_labels_from_metadata(test_metadata, "scanner_type")
    
    # Run K-NN evaluation
    knn_results_dict = knn_results(
        train_val_embeddings, train_val_disease, test_embeddings, test_disease,
        train_val_sex, test_sex, train_val_study, test_study,
        train_val_scanner, test_scanner, model_name, k=k
    )
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "Task": ["Disease Status", "Sex", "Study", "Scanner Type"],
        "K-NN Accuracy": [knn_results_dict["Disease Status"], knn_results_dict["Sex"], 
                         knn_results_dict["Study"], knn_results_dict["Scanner Type"]]
    })
    
    print("\n" + "=" * 40)
    print("K-NN RESULTS SUMMARY")
    print("=" * 40)
    print(results_df.to_string(index=False))
    
    return {
        "knn_results": knn_results_dict,
        "results_df": results_df
    }


def tsne_visualization(embeddings_dict, metadata_dict, model_name, perplexity=30):
    """Run t-SNE visualization for all classification tasks as 2x2 subplots."""
    print(f"\n{'=' * 60}")
    print(f"T-SNE VISUALIZATION: {model_name.upper()}")
    print(f"{'=' * 60}")
    
    # Combine all embeddings for visualization
    all_embeddings = np.vstack([
        embeddings_dict["train"], 
        embeddings_dict["val"], 
        embeddings_dict["test"]
    ])
    
    all_metadata = (
        metadata_dict["train"] + 
        metadata_dict["val"] + 
        metadata_dict["test"]
    )
    
    # Extract labels for visualization
    disease_labels = extract_labels_from_metadata(all_metadata, "disease_status")
    sex_labels = extract_labels_from_metadata(all_metadata, "sex")
    study_labels = extract_labels_from_metadata(all_metadata, "study")
    scanner_labels = extract_labels_from_metadata(all_metadata, "scanner_type")
    
    # Compute t-SNE once
    print(f"Computing t-SNE for {len(all_embeddings)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Create 2x2 subplot visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{model_name} - t-SNE Visualization", fontsize=16)
    
    # Plot each task
    tasks = [
        (disease_labels, "Disease Status\n(PD vs HC)", axes[0, 0]),
        (sex_labels, "Sex\n(Male vs Female)", axes[0, 1]),
        (study_labels, "Study\nClassification", axes[1, 0]),
        (scanner_labels, "Scanner Type\nClassification", axes[1, 1])
    ]
    
    for labels, title, ax in tasks:
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    return embeddings_2d


def run_linear_probe_evaluation(embeddings_dict, metadata_dict, model_name, 
                               num_epochs=100, patience=15):
    """Run linear probe evaluation on all classification tasks."""
    print(f"\n{'=' * 60}")
    print(f"LINEAR PROBE EVALUATION: {model_name.upper()}")
    print(f"{'=' * 60}")
    
    # Extract data
    train_embeddings = embeddings_dict["train"]
    val_embeddings = embeddings_dict["val"]
    test_embeddings = embeddings_dict["test"]
    
    train_metadata = metadata_dict["train"]
    val_metadata = metadata_dict["val"]
    test_metadata = metadata_dict["test"]
    
    # Extract labels for all tasks
    train_disease = extract_labels_from_metadata(train_metadata, "disease_status")
    val_disease = extract_labels_from_metadata(val_metadata, "disease_status")
    test_disease = extract_labels_from_metadata(test_metadata, "disease_status")
    
    train_sex = extract_labels_from_metadata(train_metadata, "sex")
    val_sex = extract_labels_from_metadata(val_metadata, "sex")
    test_sex = extract_labels_from_metadata(test_metadata, "sex")
    
    train_study = extract_labels_from_metadata(train_metadata, "study")
    val_study = extract_labels_from_metadata(val_metadata, "study")
    test_study = extract_labels_from_metadata(test_metadata, "study")
    
    train_scanner = extract_labels_from_metadata(train_metadata, "scanner_type")
    val_scanner = extract_labels_from_metadata(val_metadata, "scanner_type")
    test_scanner = extract_labels_from_metadata(test_metadata, "scanner_type")
    
    # Train linear probes for each task
    tasks = [
        (train_disease, val_disease, test_disease, 2, "Disease Status"),
        (train_sex, val_sex, test_sex, 2, "Sex"),
        (train_study, val_study, test_study, len(np.unique(train_study)), "Study"),
        (train_scanner, val_scanner, test_scanner, len(np.unique(train_scanner)), "Scanner Type")
    ]
    
    probe_results = {}
    trained_models = {}
    
    for train_labels, val_labels, test_labels, num_classes, task_name in tasks:
        print(f"\nTraining {task_name} linear probe...")
        
        model, val_acc, _, _, _, _ = train_linear_probe(
            train_embeddings, train_labels,
            val_embeddings, val_labels,
            task_name, num_classes, 
            num_epochs=num_epochs, patience=patience
        )
        
        probe_results[task_name] = val_acc
        trained_models[task_name] = model
        
        # Evaluate on test set
        test_acc, _ = evaluate_model_on_test_set(
            model, test_embeddings, test_labels, task_name
        )
        probe_results[f"{task_name}_test"] = test_acc
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "Task": ["Disease Status", "Sex", "Study", "Scanner Type"],
        "Validation Accuracy": [probe_results["Disease Status"], probe_results["Sex"], 
                              probe_results["Study"], probe_results["Scanner Type"]],
        "Test Accuracy": [probe_results["Disease Status_test"], probe_results["Sex_test"], 
                        probe_results["Study_test"], probe_results["Scanner Type_test"]]
    })
    
    print("\n" + "=" * 40)
    print("LINEAR PROBE RESULTS SUMMARY")
    print("=" * 40)
    print(results_df.to_string(index=False))
    
    return {
        "probe_results": probe_results,
        "trained_models": trained_models,
        "results_df": results_df
    }