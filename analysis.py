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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
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

    # Calculate F1 score - use binary for binary classification, weighted for multi-class
    if len(np.unique(y_test)) == 2:
        f1 = f1_score(y_test, y_pred, average='binary')
    else:
        f1 = f1_score(y_test, y_pred, average='weighted')

    # Print results
    print(f"K-NN (k={k}) F1 Score: {f1:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))

    return f1, y_pred


def knn_results(train_val_embeddings, train_val_disease, test_embeddings, test_disease, 
                train_val_sex, test_sex, train_val_study, test_study, 
                train_val_scanner, test_scanner, task_name, k=5):
    """Run K-NN evaluation on all classification tasks."""
    results = {}

    # Disease Status Classification (PD vs HC)
    disease_f1, disease_predictions = evaluate_knn_classification(
        train_val_embeddings, train_val_disease, test_embeddings, test_disease, 
        "Disease Status", k=k
    )
    results["Disease Status"] = disease_f1

    # Sex Classification (Male vs Female)
    sex_f1, sex_predictions = evaluate_knn_classification(
        train_val_embeddings, train_val_sex, test_embeddings, test_sex, "Sex", k=k
    )
    results["Sex"] = sex_f1

    # Study Classification
    study_f1, study_predictions = evaluate_knn_classification(
        train_val_embeddings, train_val_study, test_embeddings, test_study, "Study", k=k
    )
    results["Study"] = study_f1

    # Scanner Type Classification
    scanner_f1, scanner_predictions = evaluate_knn_classification(
        train_val_embeddings, train_val_scanner, test_embeddings, test_scanner, 
        "Scanner Type", k=k
    )
    results["Scanner Type"] = scanner_f1

    return results


def evaluate_logistic_regression_classification(X_train, y_train, X_test, y_test, task_name, 
                                               max_iter=1000, random_state=42):
    """Evaluate Logistic Regression classification performance."""
    print(f"\n=== {task_name.upper()} LOGISTIC REGRESSION CLASSIFICATION ===")

    # Train Logistic Regression classifier
    lr = LogisticRegression(max_iter=max_iter, random_state=random_state)
    lr.fit(X_train, y_train)

    # Make predictions
    y_pred = lr.predict(X_test)

    # Calculate F1 score - use binary for binary classification, weighted for multi-class
    if len(np.unique(y_test)) == 2:
        f1 = f1_score(y_test, y_pred, average='binary')
    else:
        f1 = f1_score(y_test, y_pred, average='weighted')

    # Print results
    print(f"Logistic Regression F1 Score: {f1:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))

    return f1, y_pred


def logistic_regression_results(train_val_embeddings, train_val_disease, test_embeddings, test_disease, 
                               train_val_sex, test_sex, train_val_study, test_study, 
                               train_val_scanner, test_scanner, task_name, max_iter=1000, random_state=42):
    """Run Logistic Regression evaluation on all classification tasks."""
    results = {}

    # Disease Status Classification (PD vs HC)
    disease_f1, disease_predictions = evaluate_logistic_regression_classification(
        train_val_embeddings, train_val_disease, test_embeddings, test_disease, 
        "Disease Status", max_iter=max_iter, random_state=random_state
    )
    results["Disease Status"] = disease_f1

    # Sex Classification (Male vs Female)
    sex_f1, sex_predictions = evaluate_logistic_regression_classification(
        train_val_embeddings, train_val_sex, test_embeddings, test_sex, "Sex", 
        max_iter=max_iter, random_state=random_state
    )
    results["Sex"] = sex_f1

    # Study Classification
    study_f1, study_predictions = evaluate_logistic_regression_classification(
        train_val_embeddings, train_val_study, test_embeddings, test_study, "Study", 
        max_iter=max_iter, random_state=random_state
    )
    results["Study"] = study_f1

    # Scanner Type Classification
    scanner_f1, scanner_predictions = evaluate_logistic_regression_classification(
        train_val_embeddings, train_val_scanner, test_embeddings, test_scanner, 
        "Scanner Type", max_iter=max_iter, random_state=random_state
    )
    results["Scanner Type"] = scanner_f1

    return results


def logistic_regression_evaluation(embeddings_dict, metadata_dict, model_name, 
                                  max_iter=1000, random_state=42):
    """Run Logistic Regression evaluation on all classification tasks."""
    print(f"\n{'=' * 60}")
    print(f"LOGISTIC REGRESSION EVALUATION: {model_name.upper()}")
    print(f"{'=' * 60}")
    
    # Extract data
    train_embeddings = embeddings_dict["train"]
    val_embeddings = embeddings_dict["val"]
    test_embeddings = embeddings_dict["test"]
    
    train_metadata = metadata_dict["train"]
    val_metadata = metadata_dict["val"]
    test_metadata = metadata_dict["test"]
    
    # Combine train and val for Logistic Regression (as done in notebooks)
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
    
    # Run Logistic Regression evaluation
    lr_results_dict = logistic_regression_results(
        train_val_embeddings, train_val_disease, test_embeddings, test_disease,
        train_val_sex, test_sex, train_val_study, test_study,
        train_val_scanner, test_scanner, model_name, max_iter=max_iter, random_state=random_state
    )
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "Task": ["Disease Status", "Sex", "Study", "Scanner Type"],
        "Logistic Regression F1 Score": [lr_results_dict["Disease Status"], lr_results_dict["Sex"], 
                                        lr_results_dict["Study"], lr_results_dict["Scanner Type"]]
    })
    
    print("\n" + "=" * 40)
    print("LOGISTIC REGRESSION RESULTS SUMMARY")
    print("=" * 40)
    print(results_df.to_string(index=False))
    
    return {
        "lr_results": lr_results_dict,
        "results_df": results_df
    }


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
        "K-NN F1 Score": [knn_results_dict["Disease Status"], knn_results_dict["Sex"], 
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


def calculate_metrics(y_true, y_pred):
    """Calculate F1-score, sensitivity, specificity, and accuracy."""
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')  # This is sensitivity
    
    # Calculate specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'sensitivity': recall,
        'specificity': specificity,
        'precision': precision
    }


def disease_status_subgroup_analysis(embeddings_dict, metadata_dict, model_name, 
                                   classifier_type='logistic_regression', k=5, max_iter=1000):
    """Perform subgroup analysis for disease status classification."""
    print(f"\n{'=' * 60}")
    print(f"DISEASE STATUS SUBGROUP ANALYSIS: {model_name.upper()}")
    print(f"{'=' * 60}")
    
    # Extract data
    train_embeddings = embeddings_dict["train"]
    val_embeddings = embeddings_dict["val"]
    test_embeddings = embeddings_dict["test"]
    
    train_metadata = metadata_dict["train"]
    val_metadata = metadata_dict["val"]
    test_metadata = metadata_dict["test"]
    
    # Combine train and val for training
    train_val_embeddings = np.vstack([train_embeddings, val_embeddings])
    train_val_metadata = train_metadata + val_metadata
    
    # Extract labels
    train_val_disease = extract_labels_from_metadata(train_val_metadata, "disease_status")
    test_disease = extract_labels_from_metadata(test_metadata, "disease_status")
    
    # Extract subgroup labels
    test_sex = extract_labels_from_metadata(test_metadata, "sex")
    test_scanner = extract_labels_from_metadata(test_metadata, "scanner_type")
    test_study = extract_labels_from_metadata(test_metadata, "study")
    
    # Train classifier
    if classifier_type == 'logistic_regression':
        classifier = LogisticRegression(max_iter=max_iter, random_state=42)
    elif classifier_type == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    else:
        raise ValueError("classifier_type must be 'logistic_regression' or 'knn'")
    
    classifier.fit(train_val_embeddings, train_val_disease)
    test_predictions = classifier.predict(test_embeddings)
    
    # Overall performance
    overall_metrics = calculate_metrics(test_disease, test_predictions)
    print(f"\nOverall Performance:")
    print(f"F1-Score: {overall_metrics['f1_score']:.4f}")
    print(f"Sensitivity: {overall_metrics['sensitivity']:.4f}")
    print(f"Specificity: {overall_metrics['specificity']:.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    
    # Subgroup analysis
    subgroups = {
        'Sex': test_sex,
        'Scanner Type': test_scanner,
        'Study': test_study
    }
    
    subgroup_results = {}
    
    for subgroup_name, subgroup_labels in subgroups.items():
        print(f"\n{'=' * 40}")
        print(f"SUBGROUP ANALYSIS: {subgroup_name.upper()}")
        print(f"{'=' * 40}")
        
        unique_groups = np.unique(subgroup_labels)
        subgroup_metrics = {}
        
        for group in unique_groups:
            # Get indices for this subgroup
            group_mask = subgroup_labels == group
            group_disease = test_disease[group_mask]
            group_predictions = test_predictions[group_mask]
            
            if len(np.unique(group_disease)) > 1:  # Only analyze if both classes present
                metrics = calculate_metrics(group_disease, group_predictions)
                subgroup_metrics[group] = metrics
                
                print(f"\n{subgroup_name} = {group} (n={len(group_disease)}):")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
                print(f"  Specificity: {metrics['specificity']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
            else:
                print(f"\n{subgroup_name} = {group} (n={len(group_disease)}): Skipped (only one class present)")
        
        subgroup_results[subgroup_name] = subgroup_metrics
    
    # Create summary DataFrame
    summary_data = []
    
    # Add overall results
    summary_data.append({
        'Subgroup': 'Overall',
        'Group': 'All',
        'N': len(test_disease),
        'F1-Score': overall_metrics['f1_score'],
        'Sensitivity': overall_metrics['sensitivity'],
        'Specificity': overall_metrics['specificity'],
        'Precision': overall_metrics['precision']
    })
    
    # Add subgroup results
    for subgroup_name, subgroup_metrics in subgroup_results.items():
        for group, metrics in subgroup_metrics.items():
            # Count samples in this group
            group_mask = subgroups[subgroup_name] == group
            n_samples = np.sum(group_mask)
            
            summary_data.append({
                'Subgroup': subgroup_name,
                'Group': str(group),
                'N': n_samples,
                'F1-Score': metrics['f1_score'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'Precision': metrics['precision']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n{'=' * 60}")
    print("SUBGROUP ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Disease Status Subgroup Analysis', fontsize=16)
    
    metrics_to_plot = ['F1-Score', 'Sensitivity', 'Specificity', 'Precision']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data for plotting
        plot_data = []
        plot_labels = []
        
        # Add overall
        plot_data.append(overall_metrics[metric.lower().replace('-', '_')])
        plot_labels.append('Overall')
        
        # Add subgroups
        for subgroup_name, subgroup_metrics in subgroup_results.items():
            for group, metrics in subgroup_metrics.items():
                plot_data.append(metrics[metric.lower().replace('-', '_')])
                plot_labels.append(f"{subgroup_name}\n{group}")
        
        # Create bar plot
        bars = ax.bar(range(len(plot_data)), plot_data, alpha=0.7)
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_labels, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Subgroup')
        ax.grid(True, alpha=0.3)
        
        # Color bars differently for overall vs subgroups
        bars[0].set_color('red')  # Overall in red
        for i in range(1, len(bars)):
            bars[i].set_color('skyblue')  # Subgroups in blue
    
    plt.tight_layout()
    plt.show()
    
    return {
        'overall_metrics': overall_metrics,
        'subgroup_results': subgroup_results,
        'summary_df': summary_df,
        'test_predictions': test_predictions
    }


def calculate_subgroup_f1_scores(embeddings_dict, metadata_dict, model_name):
    """Calculate F1 scores for disease classification within each sex, scanner, and study subgroup."""
    
    # Extract data
    train_embeddings = embeddings_dict["train"]
    val_embeddings = embeddings_dict["val"]
    test_embeddings = embeddings_dict["test"]
    
    train_metadata = metadata_dict["train"]
    val_metadata = metadata_dict["val"]
    test_metadata = metadata_dict["test"]
    
    # Combine train and val for training
    train_val_embeddings = np.vstack([train_embeddings, val_embeddings])
    train_val_metadata = train_metadata + val_metadata
    
    # Extract all labels
    train_val_disease = np.array([m["disease_status"] for m in train_val_metadata])
    test_disease = np.array([m["disease_status"] for m in test_metadata])
    test_sex = np.array([m["sex"] for m in test_metadata])
    test_study = np.array([m["study"] for m in test_metadata])
    test_scanner = np.array([m["scanner_type"] for m in test_metadata])
    
    # Train a single classifier on all data
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_val_embeddings, train_val_disease)
    
    # Get predictions for test set
    test_predictions = classifier.predict(test_embeddings)
    
    # Calculate overall F1 score using the same classifier and averaging method
    overall_f1 = f1_score(test_disease, test_predictions, average='binary')
    print(f"\nOverall Disease Classification F1 Score (Logistic Regression): {overall_f1:.4f}")
    
    # Calculate F1 scores for each subgroup
    results = {
        'overall_f1': overall_f1,
        'overall_predictions': test_predictions,
        'sex': {},
        'study': {},
        'scanner': {}
    }
    
    # Sex subgroups
    for sex_val in np.unique(test_sex):
        mask = test_sex == sex_val
        if len(np.unique(test_disease[mask])) > 1:  # Both classes present
            f1 = f1_score(test_disease[mask], test_predictions[mask], average='binary')
            results['sex'][sex_val] = {
                'f1': f1,
                'n_samples': np.sum(mask),
                'label': 'Female' if sex_val == 0 else 'Male'
            }
    
    # Study subgroups
    for study_val in np.unique(test_study):
        mask = test_study == study_val
        if len(np.unique(test_disease[mask])) > 1:  # Both classes present
            f1 = f1_score(test_disease[mask], test_predictions[mask], average='binary')
            results['study'][study_val] = {
                'f1': f1,
                'n_samples': np.sum(mask),
                'label': f'Study {study_val}'
            }
    
    # Scanner subgroups
    for scanner_val in np.unique(test_scanner):
        mask = test_scanner == scanner_val
        if len(np.unique(test_disease[mask])) > 1:  # Both classes present
            f1 = f1_score(test_disease[mask], test_predictions[mask], average='binary')
            results['scanner'][scanner_val] = {
                'f1': f1,
                'n_samples': np.sum(mask),
                'label': f'Scanner {scanner_val}'
            }
    
    return results


def plot_subgroup_f1_scores(checkpoint_results, random_results, model_name):
    """Create four plots showing F1 scores for overall and each sex, scanner, and study subgroup."""
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f'{model_name} - Disease Classification F1 Scores by Subgroup\n(Checkpoint vs Random)', 
                 fontsize=16, fontweight='bold')
    
    # First plot: Overall F1 score comparison
    ax_overall = axes[0]
    
    # Extract overall F1 scores
    checkpoint_overall = checkpoint_results.get('overall_f1', 0)
    random_overall = random_results.get('overall_f1', 0)
    
    # Create bar plot for overall comparison
    x_pos = [0, 1]
    heights = [checkpoint_overall, random_overall]
    colors = ['#E74C3C', 'lightgray']
    
    bars = ax_overall.bar(x_pos, heights, color=colors, alpha=0.8)
    ax_overall.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax_overall.set_title('Overall Disease Classification', fontsize=14, fontweight='bold')
    ax_overall.set_xticks(x_pos)
    ax_overall.set_xticklabels(['Checkpoint', 'Random'])
    ax_overall.grid(True, alpha=0.3, axis='y')
    ax_overall.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, f1_val in zip(bars, heights):
        height = bar.get_height()
        ax_overall.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{f1_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add improvement annotation
    improvement = checkpoint_overall - random_overall
    if abs(improvement) > 0.01:
        ax_overall.annotate(f'Δ{improvement:+.3f}', xy=(0.5, max(heights) + 0.1), 
                           fontsize=10, ha='center', fontweight='bold', 
                           color='green' if improvement > 0 else 'red')
    
    subgroups = ['sex', 'scanner', 'study']
    titles = ['Sex', 'Scanner Type', 'Study']
    colors_checkpoint = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (subgroup, title, color) in enumerate(zip(subgroups, titles, colors_checkpoint)):
        ax = axes[idx + 1]  # +1 because first axis is used for overall comparison
        
        # Get data for this subgroup
        checkpoint_data = checkpoint_results[subgroup]
        random_data = random_results[subgroup]
        
        # Prepare data for plotting
        labels = []
        checkpoint_f1s = []
        random_f1s = []
        n_samples = []
        
        for key in sorted(checkpoint_data.keys()):
            if key in random_data:  # Make sure both have this subgroup
                labels.append(checkpoint_data[key]['label'])
                checkpoint_f1s.append(checkpoint_data[key]['f1'])
                random_f1s.append(random_data[key]['f1'])
                n_samples.append(checkpoint_data[key]['n_samples'])
        
        # Create grouped bar plot
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, checkpoint_f1s, width, label='Checkpoint', 
                      color=color, alpha=0.8)
        bars2 = ax.bar(x + width/2, random_f1s, width, label='Random', 
                      color='lightgray', alpha=0.8)
        
        # Customize plot
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, f1_val in zip(bars1, checkpoint_f1s):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{f1_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar, f1_val in zip(bars2, random_f1s):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{f1_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add sample size annotations
        for i, (label, n) in enumerate(zip(labels, n_samples)):
            ax.text(i, -0.15, f'n={n}', ha='center', va='top', 
                   fontsize=8, style='italic', transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print("\n" + "=" * 80)
    print(f"SUBGROUP F1 SCORE ANALYSIS: {model_name}")
    print("=" * 80)
    
    # Print overall comparison
    checkpoint_overall = checkpoint_results.get('overall_f1', 0)
    random_overall = random_results.get('overall_f1', 0)
    overall_improvement = checkpoint_overall - random_overall
    print(f"\nOverall Disease Classification:")
    print("-" * 60)
    print(f"{'Overall (all test samples)':>25}: Checkpoint={checkpoint_overall:.4f}, "
          f"Random={random_overall:.4f}, Δ={overall_improvement:+.4f}")
    
    for subgroup, title in zip(subgroups, titles):
        print(f"\n{title} Subgroups:")
        print("-" * 60)
        checkpoint_data = checkpoint_results[subgroup]
        random_data = random_results[subgroup]
        
        for key in sorted(checkpoint_data.keys()):
            if key in random_data:
                label = checkpoint_data[key]['label']
                checkpoint_f1 = checkpoint_data[key]['f1']
                random_f1 = random_data[key]['f1']
                improvement = checkpoint_f1 - random_f1
                n = checkpoint_data[key]['n_samples']
                
                print(f"{label:>12} (n={n:3d}): Checkpoint={checkpoint_f1:.4f}, "
                      f"Random={random_f1:.4f}, Δ={improvement:+.4f}")
    
    print("=" * 80)


def plot_overall_f1_comparison(checkpoint_knn_results, random_knn_results,
                                    checkpoint_lr_results, random_lr_results, model_name):
    """Create a comprehensive plot comparing checkpoint vs random F1 scores across all tasks."""
    
    # Extract F1 score data
    tasks = ['Disease Status', 'Sex', 'Study', 'Scanner Type']
    
    # K-NN F1 scores
    knn_checkpoint = [checkpoint_knn_results['knn_results'][task] for task in tasks]
    knn_random = [random_knn_results['knn_results'][task] for task in tasks]
    
    # Logistic Regression F1 scores
    lr_checkpoint = [checkpoint_lr_results['lr_results'][task] for task in tasks]
    lr_random = [random_lr_results['lr_results'][task] for task in tasks]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} - Classification F1 Score Comparison: Checkpoint vs Random', 
                 fontsize=16, fontweight='bold')
    
    # Define colors for each task
    task_colors = ['#E74C3C', '#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Plot 1: K-NN Accuracies
    ax = axes[0]
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, knn_checkpoint, width, label='Checkpoint', alpha=0.8, color=task_colors)
    bars2 = ax.bar(x + width/2, knn_random, width, label='Random', alpha=0.8, color='lightgray')
    
    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('K-NN Classification (k=5)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha='right', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Add value labels and improvement
    for i, (bar_c, bar_r, checkpoint_val, random_val) in enumerate(zip(bars1, bars2, knn_checkpoint, knn_random)):
        # Checkpoint values
        height_c = bar_c.get_height()
        ax.text(bar_c.get_x() + bar_c.get_width()/2., height_c + 0.01,
               f'{checkpoint_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Random values
        height_r = bar_r.get_height()
        ax.text(bar_r.get_x() + bar_r.get_width()/2., height_r + 0.01,
               f'{random_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Improvement arrow
        improvement = checkpoint_val - random_val
        if abs(improvement) > 0.01:
            mid_x = x[i]
            y_start = max(height_c, height_r) + 0.08
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'Δ{improvement:+.3f}', xy=(mid_x, y_start), 
                       fontsize=9, ha='center', fontweight='bold', color=color)
    
    # Plot 2: Logistic Regression Accuracies
    ax = axes[1]
    
    bars1 = ax.bar(x - width/2, lr_checkpoint, width, label='Checkpoint', alpha=0.8, color=task_colors)
    bars2 = ax.bar(x + width/2, lr_random, width, label='Random', alpha=0.8, color='lightgray')
    
    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('Logistic Regression Classification', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha='right', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Add value labels and improvement
    for i, (bar_c, bar_r, checkpoint_val, random_val) in enumerate(zip(bars1, bars2, lr_checkpoint, lr_random)):
        # Checkpoint values
        height_c = bar_c.get_height()
        ax.text(bar_c.get_x() + bar_c.get_width()/2., height_c + 0.01,
               f'{checkpoint_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Random values
        height_r = bar_r.get_height()
        ax.text(bar_r.get_x() + bar_r.get_width()/2., height_r + 0.01,
               f'{random_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Improvement arrow
        improvement = checkpoint_val - random_val
        if abs(improvement) > 0.01:
            mid_x = x[i]
            y_start = max(height_c, height_r) + 0.08
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'Δ{improvement:+.3f}', xy=(mid_x, y_start), 
                       fontsize=9, ha='center', fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed comparison table
    print("\n" + "=" * 90)
    print("OVERALL CLASSIFICATION F1 SCORE COMPARISON")
    print("=" * 90)
    
    print("\n{:<20} {:>15} {:>15} {:>15} {:>15}".format(
        "Task", "KNN Checkpoint", "KNN Random", "LR Checkpoint", "LR Random"))
    print("-" * 90)
    
    for i, task in enumerate(tasks):
        print("{:<20} {:>15.4f} {:>15.4f} {:>15.4f} {:>15.4f}".format(
            task, knn_checkpoint[i], knn_random[i], lr_checkpoint[i], lr_random[i]))
    
    print("\n{:<20} {:>15} {:>15}".format("Task", "KNN Δ", "LR Δ"))
    print("-" * 90)
    
    for i, task in enumerate(tasks):
        knn_improvement = knn_checkpoint[i] - knn_random[i]
        lr_improvement = lr_checkpoint[i] - lr_random[i]
        print("{:<20} {:>15.4f} {:>15.4f}".format(task, knn_improvement, lr_improvement))
    
    # Calculate and print averages
    avg_knn_checkpoint = np.mean(knn_checkpoint)
    avg_knn_random = np.mean(knn_random)
    avg_lr_checkpoint = np.mean(lr_checkpoint)
    avg_lr_random = np.mean(lr_random)
    
    print("\n" + "=" * 90)
    print("AVERAGE F1 SCORES")
    print("=" * 90)
    print(f"K-NN Checkpoint:         {avg_knn_checkpoint:.4f}")
    print(f"K-NN Random:             {avg_knn_random:.4f}")
    print(f"K-NN Improvement:        {avg_knn_checkpoint - avg_knn_random:+.4f}")
    print()
    print(f"Logistic Reg Checkpoint: {avg_lr_checkpoint:.4f}")
    print(f"Logistic Reg Random:     {avg_lr_random:.4f}")
    print(f"Logistic Reg Improvement:{avg_lr_checkpoint - avg_lr_random:+.4f}")
    print("=" * 90)


def create_tsne_comparison_plots(checkpoint_embeddings, checkpoint_metadata, 
                                random_embeddings, random_metadata, model_name):
    """Create side-by-side t-SNE comparison plots."""
    
    # Combine all embeddings for both models
    checkpoint_all_embeddings = np.vstack([
        checkpoint_embeddings["train"], 
        checkpoint_embeddings["val"], 
        checkpoint_embeddings["test"]
    ])
    
    random_all_embeddings = np.vstack([
        random_embeddings["train"], 
        random_embeddings["val"], 
        random_embeddings["test"]
    ])
    
    checkpoint_all_metadata = (
        checkpoint_metadata["train"] + 
        checkpoint_metadata["val"] + 
        checkpoint_metadata["test"]
    )
    
    random_all_metadata = (
        random_metadata["train"] + 
        random_metadata["val"] + 
        random_metadata["test"]
    )
    
    # Extract labels for visualization
    checkpoint_disease_labels = np.array([m["disease_status"] for m in checkpoint_all_metadata])
    checkpoint_sex_labels = np.array([m["sex"] for m in checkpoint_all_metadata])
    checkpoint_study_labels = np.array([m["study"] for m in checkpoint_all_metadata])
    checkpoint_scanner_labels = np.array([m["scanner_type"] for m in checkpoint_all_metadata])
    
    random_disease_labels = np.array([m["disease_status"] for m in random_all_metadata])
    random_sex_labels = np.array([m["sex"] for m in random_all_metadata])
    random_study_labels = np.array([m["study"] for m in random_all_metadata])
    random_scanner_labels = np.array([m["scanner_type"] for m in random_all_metadata])
    
    # Compute t-SNE for both models
    print("Computing t-SNE for checkpoint model...")
    checkpoint_tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    checkpoint_embeddings_2d = checkpoint_tsne.fit_transform(checkpoint_all_embeddings)
    
    print("Computing t-SNE for random model...")
    random_tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    random_embeddings_2d = random_tsne.fit_transform(random_all_embeddings)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{model_name} - t-SNE Comparison: Checkpoint vs Random Initialization', 
                 fontsize=16, fontweight='bold')
    
    # Define tasks and labels
    tasks = [
        (checkpoint_disease_labels, random_disease_labels, "Disease Status\n(PD vs HC)"),
        (checkpoint_sex_labels, random_sex_labels, "Sex\n(Male vs Female)"),
        (checkpoint_study_labels, random_study_labels, "Study\nClassification"),
        (checkpoint_scanner_labels, random_scanner_labels, "Scanner Type\nClassification")
    ]
    
    for i, (checkpoint_labels, random_labels, title) in enumerate(tasks):
        # Checkpoint model plots (top row)
        ax = axes[0, i]
        scatter = ax.scatter(checkpoint_embeddings_2d[:, 0], checkpoint_embeddings_2d[:, 1], 
                           c=checkpoint_labels, cmap='viridis', alpha=0.7)
        ax.set_title(f'Checkpoint - {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        plt.colorbar(scatter, ax=ax)
        
        # Random model plots (bottom row)
        ax = axes[1, i]
        scatter = ax.scatter(random_embeddings_2d[:, 0], random_embeddings_2d[:, 1], 
                           c=random_labels, cmap='viridis', alpha=0.7)
        ax.set_title(f'Random - {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    return checkpoint_embeddings_2d, random_embeddings_2d

