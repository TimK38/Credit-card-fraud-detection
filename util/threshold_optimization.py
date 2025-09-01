import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
import warnings
warnings.filterwarnings('ignore')

def find_optimal_threshold(y_true, y_pred_proba, threshold_range=(0.01, 0.99), step=0.01, 
                          metric='f1', plot=True, figsize=(15, 10)):
    """
    Automated Threshold Optimization Function
    
    Description:
    Automatically searches for the optimal classification threshold and provides detailed visual analysis.
    Supports multiple evaluation metrics to help find the most suitable threshold for business needs.
    """

    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    results = []
    
    print("🔎 Searching for the optimal threshold...")
    print(f"➡️ Search range: {threshold_range[0]:.2f} - {threshold_range[1]:.2f}, step: {step}")
    print(f"➡️ Optimization metric: {metric.upper()}")
    print("-" * 50)
    
    for i, threshold in enumerate(thresholds):
        y_pred = (y_pred_proba >= threshold).astype(int)
        try:
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            support = np.mean(y_pred)
            
            results.append({
                'threshold': threshold,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'support': support
            })
            
            if i % 10 == 0:
                print(f"Threshold {threshold:.2f}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
                
        except Exception as e:
            print(f"Threshold {threshold:.2f} calculation error: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    
    if metric == 'f1':
        best_idx = df_results['f1_score'].idxmax()
        best_score = df_results.loc[best_idx, 'f1_score']
    elif metric == 'precision':
        best_idx = df_results['precision'].idxmax()
        best_score = df_results.loc[best_idx, 'precision']
    elif metric == 'recall':
        best_idx = df_results['recall'].idxmax()
        best_score = df_results.loc[best_idx, 'recall']
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    best_threshold = df_results.loc[best_idx, 'threshold']
    recommendations = generate_threshold_recommendations(df_results)
    
    print("\n" + "=" * 50)
    print("🎯 Optimal Threshold Search Result")
    print("=" * 50)
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Best {metric.upper()} Score: {best_score:.3f}")
    
    best_row = df_results.loc[best_idx]
    print(f"F1 Score: {best_row['f1_score']:.3f}")
    print(f"Precision: {best_row['precision']:.3f}")
    print(f"Recall: {best_row['recall']:.3f}")
    print(f"Predicted Positive Ratio: {best_row['support']:.3f}")
    
    print("\n📊 Recommended Thresholds for Different Scenarios:")
    for scenario, info in recommendations.items():
        print(f"{scenario}: {info['threshold']:.3f} (Score: {info['score']:.3f})")
    
    if plot:
        create_threshold_visualization(df_results, best_threshold, y_true, y_pred_proba, figsize)
    
    return {
        'best_threshold': best_threshold,
        'best_score': best_score,
        'all_results': df_results,
        'recommendations': recommendations,
        'best_metrics': {
            'f1_score': best_row['f1_score'],
            'precision': best_row['precision'],
            'recall': best_row['recall'],
            'support': best_row['support']
        }
    }

def generate_threshold_recommendations(df_results):
    recommendations = {}
    
    f1_best_idx = df_results['f1_score'].idxmax()
    recommendations['Best F1 Balance'] = {
        'threshold': df_results.loc[f1_best_idx, 'threshold'],
        'score': df_results.loc[f1_best_idx, 'f1_score'],
        'description': 'Balanced precision and recall'
    }
    
    precision_best_idx = df_results['precision'].idxmax()
    recommendations['Best Precision'] = {
        'threshold': df_results.loc[precision_best_idx, 'threshold'],
        'score': df_results.loc[precision_best_idx, 'precision'],
        'description': 'Minimize false positives, suitable for conservative strategy'
    }
    
    recall_best_idx = df_results['recall'].idxmax()
    recommendations['Best Recall'] = {
        'threshold': df_results.loc[recall_best_idx, 'threshold'],
        'score': df_results.loc[recall_best_idx, 'recall'],
        'description': 'Minimize false negatives, suitable for aggressive strategy'
    }
    
    high_precision = df_results[df_results['precision'] >= 0.8]
    if not high_precision.empty:
        hp_best_idx = high_precision['f1_score'].idxmax()
        recommendations['High Precision Scenario'] = {
            'threshold': high_precision.loc[hp_best_idx, 'threshold'],
            'score': high_precision.loc[hp_best_idx, 'f1_score'],
            'description': 'Best F1 with precision ≥ 80%'
        }
    
    high_recall = df_results[df_results['recall'] >= 0.8]
    if not high_recall.empty:
        hr_best_idx = high_recall['f1_score'].idxmax()
        recommendations['High Recall Scenario'] = {
            'threshold': high_recall.loc[hr_best_idx, 'threshold'],
            'score': high_recall.loc[hr_best_idx, 'f1_score'],
            'description': 'Best F1 with recall ≥ 80%'
        }
    
    return recommendations

def create_threshold_visualization(df_results, best_threshold, y_true, y_pred_proba, figsize):
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Threshold Optimization Report', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.plot(df_results['threshold'], df_results['f1_score'], 'b-', linewidth=2, label='F1 Score')
    ax1.plot(df_results['threshold'], df_results['precision'], 'r--', linewidth=2, label='Precision')
    ax1.plot(df_results['threshold'], df_results['recall'], 'g:', linewidth=2, label='Recall')
    ax1.axvline(x=best_threshold, color='orange', linestyle='-', linewidth=2, label=f'Best Threshold ({best_threshold:.3f})')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(df_results['threshold'], df_results['f1_score'], 'b-', linewidth=3)
    ax2.axvline(x=best_threshold, color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=df_results['f1_score'].max(), color='red', linestyle=':', alpha=0.7)
    ax2.fill_between(df_results['threshold'], df_results['f1_score'], alpha=0.3)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title(f'F1 Score Distribution (Best: {df_results["f1_score"].max():.3f})')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    scatter = ax3.scatter(df_results['recall'], df_results['precision'], 
                         c=df_results['threshold'], cmap='viridis', s=30, alpha=0.7)
    best_row = df_results[df_results['threshold'] == best_threshold].iloc[0]
    ax3.scatter(best_row['recall'], best_row['precision'], 
               color='red', s=100, marker='*', label=f'Best Threshold ({best_threshold:.3f})')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Tradeoff')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Threshold')
    
    ax4 = axes[1, 0]
    ax4.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    ax4.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    ax4.axvline(x=best_threshold, color='orange', linestyle='--', linewidth=2, label=f'Best Threshold ({best_threshold:.3f})')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Density')
    ax4.set_title('Predicted Probability Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    ax5.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax5.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    ax5.set_xlabel('False Positive Rate (FPR)')
    ax5.set_ylabel('True Positive Rate (TPR)')
    ax5.set_title('ROC Curve')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = axes[1, 2]
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    ax6.plot(recall_curve, precision_curve, 'g-', linewidth=2, label='PR Curve')
    ax6.axhline(y=np.mean(y_true), color='r', linestyle='--', linewidth=1, label='Baseline (Random)')
    ax6.set_xlabel('Recall')
    ax6.set_ylabel('Precision')
    ax6.set_title('Precision-Recall Curve')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nSummary Statistics:")
    print(f"Number of thresholds searched: {len(df_results)}")
    print(f"Max F1 Score: {df_results['f1_score'].max():.3f}")
    print(f"Max Precision: {df_results['precision'].max():.3f}")
    print(f"Max Recall: {df_results['recall'].max():.3f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.3f}")
