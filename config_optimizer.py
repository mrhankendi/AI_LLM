#!/usr/bin/env python3
"""
Configuration Optimizer: Find optimal (cap, batch) settings for target performance.
Given a desired tokens/s, this tool recommends the best power cap and batch size.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ConfigOptimizer:
    """
    Predicts performance and finds optimal configurations.
    """
    
    def __init__(self, data_path="moe_summary_tp4.xlsx"):
        """Load data and train the model."""
        self.df = pd.read_excel(data_path)
        self.model = None
        self.feature_cols = None
        self.X_template = None
        
        # Extract available configuration values
        self.available_caps = sorted(self.df['cap_w'].unique()) if 'cap_w' in self.df.columns else []
        self.available_batches = sorted(self.df['batch'].unique()) if 'batch' in self.df.columns else []
        self.available_tps = sorted(self.df['tp'].unique()) if 'tp' in self.df.columns else []
        self.available_models = sorted(self.df['model'].unique()) if 'model' in self.df.columns else []
        
        print(f"Available models: {self.available_models}")
        print(f"Available power caps: {self.available_caps}")
        print(f"Available batch sizes: {self.available_batches}")
        print(f"Available tensor parallelism: {self.available_tps}")
        
    def train_model(self):
        """Train a model to predict tokens/s from configuration parameters."""
        print("\n=== Training Performance Predictor ===")
        
        # Define features (exclude target and leaky columns)
        leaky_cols = [
            "avg_Tokens_per_s",
            "norm_tokens_per_s", 
            "total_gpu_power",
            "avg_gpu_power",
            "power_efficiency"
        ]
        
        # Remove noise columns
        drop_noise = ["run_id", "timestamp"]
        
        # Select features
        self.feature_cols = [c for c in self.df.columns 
                            if c not in leaky_cols and c not in drop_noise]
        
        X = self.df[self.feature_cols].copy()
        
        # One-hot encode categorical columns (like model names if included)
        X = pd.get_dummies(X, drop_first=False)
        self.X_template = X.iloc[0:1].copy()  # Save template for later
        
        # Target: tokens per second
        y = self.df["avg_Tokens_per_s"]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Model R²: {r2:.4f}")
        print(f"Model MAE: {mae:.2f} tokens/s")
        
        return self.model
    
    def predict_performance(self, cap_w, batch, model_name=None, tp=None, **other_params):
        """
        Predict tokens/s for a given configuration.
        
        Args:
            cap_w: Power cap in watts
            batch: Batch size
            model_name: Model name (e.g., 'mixtral', 'deepseek')
            tp: Tensor parallelism (optional, uses default if not provided)
            **other_params: Any other feature values needed
            
        Returns:
            Predicted tokens/s
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Create feature vector
        config = {
            'cap_w': cap_w,
            'batch': batch,
        }
        
        if model_name is not None:
            config['model'] = model_name
        elif self.available_models:
            config['model'] = self.available_models[0]  # Use first available model
        
        if tp is not None:
            config['tp'] = tp
        elif self.available_tps:
            config['tp'] = self.available_tps[0]  # Use first available TP
            
        # Add other parameters
        config.update(other_params)
        
        # Create DataFrame with one-hot encoding to match training
        X_pred = pd.DataFrame([config])
        X_pred = pd.get_dummies(X_pred, drop_first=False)
        
        # Align columns with training data
        for col in self.X_template.columns:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[self.X_template.columns]
        
        return self.model.predict(X_pred)[0]
    
    def find_optimal_config(self, target_tokens_per_s, model_name=None, tp=None, 
                           minimize_power=True, power_weight=0.5):
        """
        Find the best (cap, batch) configuration to meet a performance target.
        
        Args:
            target_tokens_per_s: Desired performance (tokens/s)
            model_name: Model name (e.g., 'mixtral', 'deepseek'). If None, uses first available.
            tp: Tensor parallelism setting (optional)
            minimize_power: If True, prefer lower power caps when performance is similar
            power_weight: Weight for power minimization (0-1, higher = prefer lower power)
            
        Returns:
            dict with optimal config and predicted performance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Set default model if not specified
        if model_name is None and self.available_models:
            model_name = self.available_models[0]
            print(f"No model specified, using: {model_name}")
        
        print(f"\n=== Finding Optimal Config for {model_name} @ {target_tokens_per_s:.1f} tokens/s ===")
        
        # Generate all possible configurations
        results = []
        
        caps = self.available_caps if self.available_caps else range(100, 301, 50)
        batches = self.available_batches if self.available_batches else [1, 4, 8, 16, 32, 64]
        tps = [tp] if tp is not None else (self.available_tps if self.available_tps else [4])
        
        for cap in caps:
            for batch_size in batches:
                for tp_val in tps:
                    pred_tokens = self.predict_performance(cap, batch_size, model_name, tp_val)
                    
                    # Score based on how close to target and power consumption
                    perf_error = abs(pred_tokens - target_tokens_per_s)
                    
                    if minimize_power:
                        # Lower score is better: balance performance error and power
                        score = perf_error * (1 - power_weight) + cap * power_weight
                    else:
                        score = perf_error
                    
                    results.append({
                        'model': model_name,
                        'cap_w': cap,
                        'batch': batch_size,
                        'tp': tp_val,
                        'predicted_tokens_per_s': pred_tokens,
                        'error': perf_error,
                        'score': score
                    })
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score')
        
        # Get top recommendations
        top_configs = results_df.head(5)
        
        print("\n📊 Top 5 Recommended Configurations:")
        print("="*80)
        for idx, row in top_configs.iterrows():
            print(f"Model: {row['model']:10s} | Cap: {row['cap_w']:3.0f}W | Batch: {row['batch']:3.0f} | TP: {row['tp']:1.0f} | "
                  f"Predicted: {row['predicted_tokens_per_s']:7.1f} tokens/s | "
                  f"Error: {row['error']:6.1f}")
        
        best = results_df.iloc[0]
        print(f"\n✅ BEST: model={best['model']}, cap_w={best['cap_w']:.0f}, batch={best['batch']:.0f}, tp={best['tp']:.0f}")
        print(f"   Expected: {best['predicted_tokens_per_s']:.1f} tokens/s "
              f"(target: {target_tokens_per_s:.1f})")
        
        return {
            'model': best['model'],
            'cap_w': int(best['cap_w']),
            'batch': int(best['batch']),
            'tp': int(best['tp']),
            'predicted_tokens_per_s': best['predicted_tokens_per_s'],
            'all_results': results_df
        }
    
    def visualize_config_space(self, model_name=None, tp=None, save_path="config_optimization_heatmap.png"):
        """
        Create a heatmap showing predicted performance across cap/batch combinations.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if model_name is None and self.available_models:
            model_name = self.available_models[0]
        
        caps = self.available_caps if self.available_caps else range(100, 301, 50)
        batches = self.available_batches if self.available_batches else [1, 4, 8, 16, 32, 64]
        tp_val = tp if tp is not None else (self.available_tps[0] if self.available_tps else 4)
        
        # Create grid
        perf_grid = np.zeros((len(batches), len(caps)))
        
        for i, batch_size in enumerate(batches):
            for j, cap in enumerate(caps):
                perf_grid[i, j] = self.predict_performance(cap, batch_size, model_name, tp_val)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            perf_grid,
            xticklabels=[f"{c}W" for c in caps],
            yticklabels=[f"B{b}" for b in batches],
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Tokens/s'}
        )
        plt.xlabel('Power Cap (W)')
        plt.ylabel('Batch Size')
        plt.title(f'Predicted Performance Heatmap - {model_name} (TP={tp_val})')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📈 Heatmap saved to {save_path}")
        plt.show()
    
    def save_model(self, path="models/config_optimizer.pkl"):
        """Save trained model to disk."""
        Path(path).parent.mkdir(exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'X_template': self.X_template,
            'available_caps': self.available_caps,
            'available_batches': self.available_batches,
            'available_tps': self.available_tps,
            'available_models': self.available_models
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path="models/config_optimizer.pkl"):
        """Load trained model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_cols = data['feature_cols']
        self.X_template = data['X_template']
        self.available_caps = data['available_caps']
        self.available_batches = data['available_batches']
        self.available_tps = data['available_tps']
        self.available_models = data.get('available_models', [])
        print(f"✅ Model loaded from {path}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = ConfigOptimizer("moe_summary_tp4.xlsx")
    
    # Train the model
    optimizer.train_model()
    
    # Example 1: Find config for mixtral @ 400 tokens/s
    target_tps = 400.0
    model_name = "mixtral"
    
    result = optimizer.find_optimal_config(
        target_tokens_per_s=target_tps,
        model_name=model_name,
        minimize_power=True,
        power_weight=0.3  # Prefer performance over power savings
    )
    
    # Example 2: Find config for deepseek @ 2000 tokens/s, prioritize low power
    result2 = optimizer.find_optimal_config(
        target_tokens_per_s=2000,
        model_name="deepseek",
        minimize_power=True,
        power_weight=1.0  # Strongly prefer lower power
    )
    
    # Example 3: Visualize the configuration space for mixtral
    optimizer.visualize_config_space(model_name="mixtral")
    
    # Save model for later use
    optimizer.save_model()
    
    # Example of loading and using saved model
    # optimizer2 = ConfigOptimizer("moe_summary_tp4.xlsx")
    # optimizer2.load_model()
    # result = optimizer2.find_optimal_config(1500)
