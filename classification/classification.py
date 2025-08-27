import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  # Changed from SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TextClassifier:
    def __init__(self):
        print("\n[INFO] Initializing Text Classifier...")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = MultinomialNB()  # Changed to Naive Bayes classifier
        self.labels = []  # Initialize as empty list
        
    def load_data(self, filepath):
        """Load and prepare data from CSV file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"CSV file not found at: {filepath}")
                
            print(f"\n[INFO] Loading data from {filepath}")
            df = pd.read_csv(filepath)
            
            if df.empty:
                raise ValueError("The CSV file is empty")
                
            if 'title' not in df.columns or 'summary' not in df.columns or 'category' not in df.columns:
                raise ValueError("CSV file must contain 'title', 'summary', and 'category' columns")
                
            df['text'] = df['title'] + ' ' + df['summary']
            print(f"[INFO] Loaded {len(df)} articles")
            return df['text'], df['category']
            
        except Exception as e:
            print(f"\n[ERROR] Failed to load data: {str(e)}")
            raise

    def prepare_data(self, X, y):
        """Convert text to TF-IDF features and split data"""
        print("\n[INFO] Converting text to TF-IDF features...")
        X_transformed = self.vectorizer.fit_transform(X)
        print(f"[INFO] Created {X_transformed.shape[1]} features")
        
        # Store unique labels in sorted order
        self.labels = sorted(y.unique())
        print(f"[INFO] Found {len(self.labels)} unique categories: {', '.join(self.labels)}")
        
        print("\n[INFO] Splitting data into training and test sets...")
        return X_transformed, self.labels
    
    def train(self, X_train, y_train):
        """Train the Naive Bayes classifier"""
        print(f"\n[INFO] Training Naive Bayes classifier on {X_train.shape[0]} samples...")
        self.classifier.fit(X_train, y_train)
        print("[INFO] Training completed")
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return predictions and metrics"""
        print(f"\n[INFO] Evaluating model on {X_test.shape[0]} test samples...")
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[INFO] Test accuracy: {accuracy*100:.2f}%")
        return y_pred, accuracy
    
    def get_full_dataset_confusion_matrix(self, X_full, y_full):
        """Get confusion matrix for the full dataset"""
        print("\n[INFO] Generating confusion matrix for full dataset...")
        y_pred_full = self.classifier.predict(X_full)
        return confusion_matrix(y_full, y_pred_full)
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        print("\n[INFO] Generating confusion matrix plot...")
        if self.labels is None or len(self.labels) == 0:
            print("[WARNING] Labels not found, using numerical indices")
            self.labels = list(range(cm.shape[0]))
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.labels,
                   yticklabels=self.labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        
    def classify_text(self, text):
        """Classify a single text input with confidence score"""
        print("\n[INFO] Classifying input text...")
        
        # Transform the input text using the same vectorizer
        X_input = self.vectorizer.transform([text])
        
        # Get prediction probabilities directly from Naive Bayes
        try:
            probabilities = self.classifier.predict_proba(X_input)[0]
            
            # Get the prediction and its confidence
            prediction_idx = probabilities.argmax()
            confidence = probabilities[prediction_idx]
            prediction = self.labels[prediction_idx]
            
            print(f"[INFO] Predicted category: {prediction} (confidence: {confidence:.2%})")
            print("\nConfidence scores for all categories:")
            for label, prob in zip(self.labels, probabilities):
                print(f"{label}: {prob:.2%}")
                
            return prediction, confidence
            
        except Exception as e:
            print(f"[ERROR] Classification error: {str(e)}")
            return None, 0.0

class TextClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Classification System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self.classifier = TextClassifier()
        self.setup_ui()
        
    def setup_ui(self):
        # Create main containers
        left_frame = ttk.Frame(self.root, padding="10")
        right_frame = ttk.Frame(self.root, padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Training section
        train_frame = ttk.LabelFrame(left_frame, text="Model Training", padding="10")
        train_frame.pack(fill=tk.X, pady=5)
        
        self.train_button = ttk.Button(train_frame, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=5)
        
        # Metrics section
        metrics_frame = ttk.LabelFrame(left_frame, text="Model Metrics", padding="10")
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, height=10, width=50)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # Classification section
        classify_frame = ttk.LabelFrame(right_frame, text="Text Classification", padding="10")
        classify_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(classify_frame, text="Enter text to classify:").pack(pady=5)
        self.input_text = scrolledtext.ScrolledText(classify_frame, height=5, width=50)
        self.input_text.pack(fill=tk.X, pady=5)
        
        self.classify_button = ttk.Button(classify_frame, text="Classify Text", 
                                        command=self.classify_text)
        self.classify_button.pack(pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(classify_frame, text="Classification Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8, width=50)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Confusion Matrix plot
        self.plot_frame = ttk.LabelFrame(right_frame, text="Confusion Matrix", padding="10")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Style configuration
        style = ttk.Style()
        style.configure('TButton', padding=6)
        style.configure('TLabelframe', background='#f0f0f0')
        style.configure('TLabelframe.Label', font=('Helvetica', 10, 'bold'))
        
    def train_model(self):
        try:
            self.train_button.config(state='disabled')
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "Training model, please wait...\n")
            
            # Load and prepare data
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'classification_data.csv')
            X, y = self.classifier.load_data(data_path)
            X_transformed, labels = self.classifier.prepare_data(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
            
            # Train and evaluate
            self.classifier.train(X_train, y_train)
            y_pred, accuracy = self.classifier.evaluate(X_test, y_test)
            
            # Display metrics
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, f"Accuracy: {accuracy*100:.2f}%\n\n")
            self.metrics_text.insert(tk.END, "Classification Report:\n")
            self.metrics_text.insert(tk.END, classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            cm = self.classifier.get_full_dataset_confusion_matrix(X_transformed, y)
            self.plot_confusion_matrix(cm)
            
            messagebox.showinfo("Success", "Model training completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.train_button.config(state='normal')
    
    def classify_text(self):
        text = self.input_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to classify")
            return
            
        try:
            prediction, confidence = self.classifier.classify_text(text)
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Predicted Category: {prediction}\n")
            self.results_text.insert(tk.END, f"Confidence: {confidence:.2%}\n\n")
            self.results_text.insert(tk.END, "Confidence scores for all categories:\n")
            for label, prob in zip(self.classifier.labels, 
                                 self.classifier.classifier.predict_proba([self.classifier.vectorizer.transform([text]).toarray()[0]])[0]):
                self.results_text.insert(tk.END, f"{label}: {prob:.2%}\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"Classification error: {str(e)}")
    
    def plot_confusion_matrix(self, cm):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classifier.labels,
                   yticklabels=self.classifier.labels, ax=ax)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close()

def main():
    root = tk.Tk()
    app = TextClassifierUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
