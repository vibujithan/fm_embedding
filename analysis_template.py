"""
Template for running complete checkpoint vs random model analysis.
Import this in notebooks and call run_complete_analysis().

This refactored version eliminates duplicate computations:
- t-SNE is computed only once (in create_tsne_comparison_plots)
- Subgroup F1 scores are calculated once per model
"""

import numpy as np
import torch
from analysis import (
    extract_embeddings, knn_evaluation, logistic_regression_evaluation,
    calculate_subgroup_f1_scores, plot_subgroup_f1_scores,
    plot_overall_f1_comparison, create_tsne_comparison_plots
)


def run_complete_analysis(checkpoint_model, random_model, train_loader, val_loader, 
                          test_loader, device, model_name, skip_tsne_individual=True):
    """
    Run complete analysis comparing checkpoint and random models.
    
    Args:
        checkpoint_model: Model loaded from checkpoint
        random_model: Randomly initialized model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: torch device
        model_name: Name for labeling plots
        skip_tsne_individual: Skip individual t-SNE plots (default True to avoid duplication)
    
    Returns:
        dict: Results dictionary with checkpoint and random model evaluations
    """
    
    print("=" * 80)
    print(f"RUNNING COMPLETE ANALYSIS FOR {model_name}")
    print("=" * 80)
    
    # ========================================
    # STEP 1: Extract embeddings for both models
    # ========================================
    print("\n" + "=" * 80)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 80)
    
    print("\nCheckpoint model:")
    train_emb, train_meta = extract_embeddings(checkpoint_model, device, train_loader)
    val_emb, val_meta = extract_embeddings(checkpoint_model, device, val_loader)
    test_emb, test_meta = extract_embeddings(checkpoint_model, device, test_loader)
    
    embeddings = {"train": train_emb, "val": val_emb, "test": test_emb}
    metadata = {"train": train_meta, "val": val_meta, "test": test_meta}
    
    print("\nRandom model:")
    random_train_emb, random_train_meta = extract_embeddings(random_model, device, train_loader)
    random_val_emb, random_val_meta = extract_embeddings(random_model, device, val_loader)
    random_test_emb, random_test_meta = extract_embeddings(random_model, device, test_loader)
    
    random_embeddings = {"train": random_train_emb, "val": random_val_emb, "test": random_test_emb}
    random_metadata = {"train": random_train_meta, "val": random_val_meta, "test": random_test_meta}
    
    # ========================================
    # STEP 2: Run K-NN and Logistic Regression evaluations
    # ========================================
    print("\n" + "=" * 80)
    print("RUNNING CLASSIFICATION EVALUATIONS")
    print("=" * 80)
    
    print("\nCheckpoint model:")
    knn_results = knn_evaluation(embeddings, metadata, model_name, k=5)
    lr_results = logistic_regression_evaluation(embeddings, metadata, model_name, 
                                                max_iter=1000, random_state=42)
    
    print("\nRandom model:")
    random_knn_results = knn_evaluation(random_embeddings, random_metadata, 
                                        f"Random_{model_name}", k=5)
    random_lr_results = logistic_regression_evaluation(random_embeddings, random_metadata, 
                                                      f"Random_{model_name}", 
                                                      max_iter=1000, random_state=42)
    
    # ========================================
    # STEP 3: Calculate subgroup F1 scores (once per model)
    # ========================================
    print("\n" + "=" * 80)
    print("CALCULATING SUBGROUP F1 SCORES")
    print("=" * 80)
    
    checkpoint_subgroup_results = calculate_subgroup_f1_scores(embeddings, metadata, 
                                                               f"{model_name}_Checkpoint")
    random_subgroup_results = calculate_subgroup_f1_scores(random_embeddings, random_metadata, 
                                                           f"{model_name}_Random")
    
    # ========================================
    # STEP 4: Generate comparison plots
    # ========================================
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    
    # Plot 1: Subgroup F1 scores (sex, scanner, study)
    print("\n[1/3] Generating subgroup F1 score plots...")
    plot_subgroup_f1_scores(checkpoint_subgroup_results, random_subgroup_results, model_name)
    
    # Plot 2: Overall F1 score comparison (K-NN and LR)
    print("\n[2/3] Generating overall F1 score comparison plots...")
    plot_overall_f1_comparison(knn_results, random_knn_results, 
                                    lr_results, random_lr_results, model_name)
    
    # Plot 3: t-SNE comparison (computed only once here, not in individual evaluations)
    print("\n[3/3] Generating t-SNE comparison plots...")
    checkpoint_tsne_2d, random_tsne_2d = create_tsne_comparison_plots(
        embeddings, metadata, random_embeddings, random_metadata, model_name
    )
    
    # ========================================
    # Done!
    # ========================================
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated outputs:")
    print("  • Subgroup F1 score plots (sex, scanner, study)")
    print("  • Overall F1 score comparison (K-NN and Logistic Regression)")
    print("  • t-SNE comparison visualizations")
    print("=" * 80)
    
    return {
        'checkpoint': {
            'embeddings': embeddings,
            'metadata': metadata,
            'knn_results': knn_results,
            'lr_results': lr_results,
            'tsne_2d': checkpoint_tsne_2d,
            'subgroup_results': checkpoint_subgroup_results
        },
        'random': {
            'embeddings': random_embeddings,
            'metadata': random_metadata,
            'knn_results': random_knn_results,
            'lr_results': random_lr_results,
            'tsne_2d': random_tsne_2d,
            'subgroup_results': random_subgroup_results
        }
    }

