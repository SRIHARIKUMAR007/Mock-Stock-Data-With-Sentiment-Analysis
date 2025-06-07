
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DiabetesPredictionAnalyzer:
    """
    A comprehensive diabetes prediction analyzer using multiple ML algorithms.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {
            'Algorithm': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'Support': []
        }
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self, file_path=None):
        """
        Load diabetes dataset from file or prompt for upload in Colab.
        """
        if file_path and os.path.exists(file_path):
            self.df = pd.read_csv(file_path)
            print(f"Dataset loaded from {file_path}")
        else:
            # For Google Colab environment
            try:
                print("Upload your diabetes dataset CSV file:")
                from google.colab import files
                uploaded = files.upload()
                filename = next(iter(uploaded))
                self.df = pd.read_csv(filename)
                print(f"Dataset uploaded successfully: {filename}")
            except ImportError:
                # For local environment
                file_path = input("Enter the path to your diabetes dataset CSV file: ")
                self.df = pd.read_csv(file_path)
                print(f"Dataset loaded from {file_path}")
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Column names: {self.df.columns.tolist()}")
        return self.df
    
    def preprocess_data(self):
        """
        Comprehensive data preprocessing pipeline.
        """
        print("\n" + "="*50)
        print("STARTING DATA PREPROCESSING")
        print("="*50)
        
        # Replace empty strings with NaN
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        
        # Convert columns to numeric
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"Missing values before preprocessing:\n{self.df.isnull().sum()}")
        
        # Identify target column
        self.target_col = self._identify_target_column()
        print(f"Target column identified: '{self.target_col}'")
        
        # Handle zero values (replace with NaN where medically impossible)
        self._handle_zero_values()
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print("Data preprocessing completed successfully!")
        
        return X_scaled, y
    
    def _identify_target_column(self):
        """
        Identify the target column from common naming conventions.
        """
        possible_targets = ['Class', 'Outcome', 'Target', 'Diagnosis', 'Label', 'Result']
        
        for col in possible_targets:
            if col in self.df.columns:
                return col
        
        # If no common name found, use the last column
        return self.df.columns[-1]
    
    def _handle_zero_values(self):
        """
        Replace zero values with NaN where medically impossible.
        """
        # Columns that can legitimately have zero values
        valid_zero_cols = ['Pregnancies', 'Age', self.target_col]
        if 'ID' in self.df.columns:
            valid_zero_cols.append('ID')
        
        # Replace zeros in other columns
        zero_invalid_cols = [col for col in self.df.columns if col not in valid_zero_cols]
        self.df[zero_invalid_cols] = self.df[zero_invalid_cols].replace(0, np.nan)
        
        print(f"Zero values replaced with NaN in columns: {zero_invalid_cols}")
    
    def train_models(self):
        """
        Train all machine learning models and evaluate performance.
        """
        print("\n" + "="*50)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*50)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\n{name}:")
            print("-" * 30)
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            # Get detailed classification report
            report = classification_report(self.y_test, y_pred, output_dict=True)
            weighted_avg = report['weighted avg']
            
            # Store results
            self.results['Algorithm'].append(name)
            self.results['Accuracy'].append(accuracy)
            self.results['Precision'].append(precision)
            self.results['Recall'].append(weighted_avg['recall'])
            self.results['F1-Score'].append(weighted_avg['f1-score'])
            self.results['Support'].append(weighted_avg['support'])
            
            # Store model and predictions
            self.models[name] = {
                'model': model,
                'predictions': y_pred,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'accuracy': accuracy,
                'precision': precision
            }
            
            # Print metrics
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {weighted_avg['recall']:.4f}")
            print(f"F1-Score: {weighted_avg['f1-score']:.4f}")
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations for analysis.
        """
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # 1. Algorithm Performance Comparison
        self._plot_algorithm_comparison()
        
        # 2. Feature Correlation Heatmap
        self._plot_correlation_heatmap()
        
        # 3. Random Forest Feature Importance
        if 'Random Forest' in self.models:
            self._plot_feature_importance()
        
        # 4. Confusion Matrix for Best Model
        self._plot_best_model_confusion_matrix()
        
        print("All visualizations saved to 'results' directory!")
    
    def _plot_algorithm_comparison(self):
        """
        Create algorithm performance comparison chart.
        """
        results_df = pd.DataFrame(self.results).sort_values('Accuracy', ascending=False)
        
        plt.figure(figsize=(14, 8))
        
        # Create bar positions
        bar_width = 0.35
        x = np.arange(len(results_df['Algorithm']))
        
        # Create bars
        accuracy_bars = plt.bar(x - bar_width/2, results_df['Accuracy'], bar_width, 
                               color='#2E86AB', label='Accuracy', alpha=0.8)
        precision_bars = plt.bar(x + bar_width/2, results_df['Precision'], bar_width, 
                                color='#A23B72', label='Precision', alpha=0.8)
        
        # Add value labels on bars
        for i, bar in enumerate(accuracy_bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for i, bar in enumerate(precision_bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Algorithm Performance Comparison\n(Accuracy vs Precision)', 
                 fontsize=14, fontweight='bold')
        plt.xticks(x, results_df['Algorithm'], rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.legend(fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self):
        """
        Create feature correlation heatmap.
        """
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        correlation = self.df.drop(columns=[self.target_col]).corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        
        # Create heatmap
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   mask=mask, center=0, square=True, cbar_kws={'shrink': 0.8})
        
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self):
        """
        Plot Random Forest feature importance.
        """
        rf_model = self.models['Random Forest']['model']
        
        plt.figure(figsize=(10, 8))
        
        # Get feature importance
        importance = pd.Series(rf_model.feature_importances_, index=self.feature_names)
        importance = importance.sort_values(ascending=False)
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
        bars = plt.barh(importance.index, importance.values, color=colors)
        
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_best_model_confusion_matrix(self):
        """
        Plot confusion matrix for the best performing model.
        """
        # Find best model by accuracy
        best_model_name = max(self.models.keys(), 
                            key=lambda x: self.models[x]['accuracy'])
        
        plt.figure(figsize=(8, 6))
        
        cm = self.models[best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {best_model_name}\n(Best Performing Model)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, filename='diabetes_analysis_results.xlsx'):
        """
        Save all results to Excel file.
        """
        print(f"\nSaving results to {filename}...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Original dataset
            self.df.to_excel(writer, sheet_name='Original_Data', index=False)
            
            # Processed dataset
            processed_df = pd.concat([
                pd.DataFrame(self.X_train.values, columns=self.feature_names),
                pd.DataFrame(self.X_test.values, columns=self.feature_names)
            ], ignore_index=True)
            processed_df.to_excel(writer, sheet_name='Processed_Data', index=False)
            
            # Algorithm comparison
            results_df = pd.DataFrame(self.results)
            results_df.to_excel(writer, sheet_name='Algorithm_Comparison', index=False)
            
            # Individual model predictions
            for name, model_data in self.models.items():
                predictions_df = pd.DataFrame({
                    'Actual': self.y_test,
                    'Predicted': model_data['predictions']
                })
                sheet_name = f"{name.replace(' ', '_')}_Predictions"
                predictions_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Results saved successfully to {filename}")
        
        # Download file in Google Colab
        try:
            from google.colab import files
            files.download(filename)
            print("File downloaded automatically!")
        except ImportError:
            print(f"File saved locally: {filename}")
    
    def print_summary(self):
        """
        Print comprehensive analysis summary.
        """
        print("\n" + "="*60)
        print("DIABETES PREDICTION ANALYSIS SUMMARY")
        print("="*60)
        
        results_df = pd.DataFrame(self.results).sort_values('Accuracy', ascending=False)
        
        print(f"\n📊 Dataset Information:")
        print(f"   • Total samples: {len(self.df)}")
        print(f"   • Features: {len(self.feature_names)}")
        print(f"   • Training samples: {len(self.X_train)}")
        print(f"   • Test samples: {len(self.X_test)}")
        
        print(f"\n🏆 Best Performing Algorithm:")
        best_row = results_df.iloc[0]
        print(f"   • Algorithm: {best_row['Algorithm']}")
        print(f"   • Accuracy: {best_row['Accuracy']:.4f}")
        print(f"   • Precision: {best_row['Precision']:.4f}")
        print(f"   • F1-Score: {best_row['F1-Score']:.4f}")
        
        print(f"\n📈 All Algorithm Rankings:")
        for i, row in results_df.iterrows():
            print(f"   {results_df.index.get_loc(i)+1}. {row['Algorithm']}: "
                  f"{row['Accuracy']:.4f} accuracy")
        
        print(f"\n📁 Generated Files:")
        print(f"   • Excel Report: diabetes_analysis_results.xlsx")
        print(f"   • Visualizations: results/ directory")
        print(f"   • Algorithm comparison chart")
        print(f"   • Feature correlation heatmap")
        print(f"   • Feature importance plot")
        print(f"   • Confusion matrix")
        
        print(f"\n✅ Analysis completed successfully!")


def main():
    """
    Main function to run the diabetes prediction analysis.
    """
    print("🔬 DIABETES PREDICTION ANALYSIS")
    print("Using Multiple Data Mining Algorithms")
    print("-" * 50)
    
    # Initialize analyzer
    analyzer = DiabetesPredictionAnalyzer()
    
    try:
        # Load data
        analyzer.load_data()
        
        # Preprocess data
        analyzer.preprocess_data()
        
        # Train models
        analyzer.train_models()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        analyzer.save_results()
        
        # Print summary
        analyzer.print_summary()
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        print("Please check your dataset format and try again.")


if __name__ == "__main__":
    main()