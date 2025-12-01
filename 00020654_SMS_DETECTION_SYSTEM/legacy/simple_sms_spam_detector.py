#!/usr/bin/env python3
"""
Simple SMS Spam Detection System
Works with basic Python libraries
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

class SimpleSMSSpamDetector:
    """
    Simple SMS Spam Detection System
    """
    
    def __init__(self):
        """Initialize the detector"""
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                          'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 
                          'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 
                          'does', 'did', 'will', 'would', 'could', 'should', 'i', 'you',
                          'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        self.best_model = None
        self.best_model_name = ""
        
    def load_data(self, file_path="spam (1).csv"):
        """
        Load the SMS dataset
        """
        print("Loading SMS dataset...")
        
        try:
            # Try different encodings
            for encoding in ['latin1', 'utf-8', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read file with any encoding")
            
            print(f"Dataset loaded with {encoding} encoding")
            
            # Clean the dataset
            df = df.iloc[:, :2]  # Keep only first two columns
            df.columns = ['label', 'message']
            df = df.dropna()  # Remove missing values
            
            # Convert labels to binary
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            
            print(f"Dataset Statistics:")
            print(f"   Total messages: {len(df):,}")
            print(f"   Ham messages: {sum(df['label'] == 0):,}")
            print(f"   Spam messages: {sum(df['label'] == 1):,}")
            print(f"   Spam percentage: {(sum(df['label'] == 1)/len(df)*100):.1f}%")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_text(self, text):
        """
        Basic text preprocessing
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', '', text)
        
        # Remove extra whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def extract_features(self, df):
        """
        Extract comprehensive features from messages
        """
        print("Extracting features...")
        
        # Basic text features
        df['char_count'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()
        df['avg_word_length'] = df['message'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        
        # Punctuation features
        df['exclamation_count'] = df['message'].str.count('!')
        df['question_count'] = df['message'].str.count(r'\?')
        df['capital_count'] = df['message'].str.count(r'[A-Z]')
        df['digit_count'] = df['message'].str.count(r'\d')
        
        # Spam-specific features
        currency_pattern = r'[$£€¥₹]|\b(money|cash|prize|win|free|offer|discount)\b'
        df['financial_words'] = df['message'].str.count(currency_pattern, flags=re.IGNORECASE)
        
        urgency_pattern = r'\b(urgent|asap|immediately|now|hurry|limited|expire|act)\b'
        df['urgency_words'] = df['message'].str.count(urgency_pattern, flags=re.IGNORECASE)
        
        action_pattern = r'\b(click|call|text|reply|send|subscribe|buy|order)\b'
        df['action_words'] = df['message'].str.count(action_pattern, flags=re.IGNORECASE)
        
        # Contact features
        df['phone_numbers'] = df['message'].str.count(r'\b\d{5,}\b')
        df['urls'] = df['message'].str.count(r'http|www|\.com|\.org|\.net')
        
        print("Feature extraction completed!")
        return df
    
    def visualize_dataset(self, df):
        """
        Create comprehensive dataset visualizations
        """
        print("Creating dataset visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SMS Spam Detection - Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Label distribution
        labels = ['Ham', 'Spam']
        counts = df['label'].value_counts().sort_index()
        colors = ['#2ecc71', '#e74c3c']
        
        axes[0, 0].bar(labels, counts, color=colors, alpha=0.8)
        axes[0, 0].set_title('Message Distribution')
        axes[0, 0].set_ylabel('Count')
        for i, count in enumerate(counts):
            axes[0, 0].text(i, count + 50, str(count), ha='center', fontweight='bold')
        
        # 2. Message length distribution
        df.boxplot(column='char_count', by='label', ax=axes[0, 1])
        axes[0, 1].set_title('Message Length by Type')
        axes[0, 1].set_xlabel('Label (0=Ham, 1=Spam)')
        axes[0, 1].set_ylabel('Character Count')
        
        # 3. Word count distribution
        df.boxplot(column='word_count', by='label', ax=axes[0, 2])
        axes[0, 2].set_title('Word Count by Type')
        axes[0, 2].set_xlabel('Label (0=Ham, 1=Spam)')
        axes[0, 2].set_ylabel('Word Count')
        
        # 4. Financial words
        spam_financial = df[df['label'] == 1]['financial_words'].mean()
        ham_financial = df[df['label'] == 0]['financial_words'].mean()
        axes[1, 0].bar(['Ham', 'Spam'], [ham_financial, spam_financial], 
                      color=colors, alpha=0.8)
        axes[1, 0].set_title('Avg Financial Words')
        axes[1, 0].set_ylabel('Average Count')
        
        # 5. Urgency words
        spam_urgency = df[df['label'] == 1]['urgency_words'].mean()
        ham_urgency = df[df['label'] == 0]['urgency_words'].mean()
        axes[1, 1].bar(['Ham', 'Spam'], [ham_urgency, spam_urgency], 
                      color=colors, alpha=0.8)
        axes[1, 1].set_title('Avg Urgency Words')
        axes[1, 1].set_ylabel('Average Count')
        
        # 6. Exclamation marks
        spam_excl = df[df['label'] == 1]['exclamation_count'].mean()
        ham_excl = df[df['label'] == 0]['exclamation_count'].mean()
        axes[1, 2].bar(['Ham', 'Spam'], [ham_excl, spam_excl], 
                      color=colors, alpha=0.8)
        axes[1, 2].set_title('Avg Exclamation Marks')
        axes[1, 2].set_ylabel('Average Count')
        
        plt.tight_layout()
        plt.show()
        
        # Feature correlation heatmap
        plt.figure(figsize=(12, 8))
        feature_cols = ['char_count', 'word_count', 'exclamation_count', 'capital_count',
                       'financial_words', 'urgency_words', 'action_words', 'phone_numbers']
        correlation_matrix = df[feature_cols + ['label']].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("Dataset visualizations completed!")
        return df
    
    def analyze_data(self, df):
        """
        Basic data analysis
        """
        print("\n" + "="*60)
        print("DATA ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nMessage Length Analysis:")
        print(f"Average Ham length: {df[df['label']==0]['char_count'].mean():.1f} characters")
        print(f"Average Spam length: {df[df['label']==1]['char_count'].mean():.1f} characters")
        
        print(f"\nWord Count Analysis:")
        print(f"Average Ham words: {df[df['label']==0]['word_count'].mean():.1f}")
        print(f"Average Spam words: {df[df['label']==1]['word_count'].mean():.1f}")
        
        print(f"\nSpam Indicators:")
        print(f"Financial words - Ham avg: {df[df['label']==0]['financial_words'].mean():.2f}, Spam avg: {df[df['label']==1]['financial_words'].mean():.2f}")
        print(f"Urgency words - Ham avg: {df[df['label']==0]['urgency_words'].mean():.2f}, Spam avg: {df[df['label']==1]['urgency_words'].mean():.2f}")
        print(f"Exclamations - Ham avg: {df[df['label']==0]['exclamation_count'].mean():.2f}, Spam avg: {df[df['label']==1]['exclamation_count'].mean():.2f}")
        
        # Show some examples
        print(f"\nSample SPAM messages:")
        spam_samples = df[df['label']==1]['message'].head(3)
        for i, msg in enumerate(spam_samples, 1):
            print(f"  {i}. {msg}")
        
        print(f"\nSample HAM messages:")
        ham_samples = df[df['label']==0]['message'].head(3)
        for i, msg in enumerate(ham_samples, 1):
            print(f"  {i}. {msg}")
    
    def train_models(self, df):
        """
        Train multiple machine learning models
        """
        print("\nTraining machine learning models...")
        
        # Prepare text features
        df['cleaned_message'] = df['message'].apply(self.preprocess_text)
        X_text = self.vectorizer.fit_transform(df['cleaned_message'])
        
        # Prepare numerical features
        numerical_features = ['char_count', 'word_count', 'avg_word_length', 
                            'exclamation_count', 'question_count', 'capital_count',
                            'digit_count', 'financial_words', 'urgency_words', 
                            'action_words', 'phone_numbers', 'urls']
        
        X_numerical = df[numerical_features].values
        
        # Combine features
        X_combined = hstack([X_text, X_numerical])
        y = df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store for later use
        self.X_test, self.y_test = X_test, y_test
        
        # Define models
        models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        best_score = 0
        
        print("\n" + "="*60)
        print("MODEL TRAINING RESULTS")
        print("="*60)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Cross-Validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Track best model
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} (CV: {best_score:.4f})")
        
        # Detailed report for best model
        print(f"\nDETAILED CLASSIFICATION REPORT - {self.best_model_name}:")
        print("-" * 60)
        best_pred = results[self.best_model_name]['predictions']
        print(classification_report(y_test, best_pred, target_names=['Ham', 'Spam']))
        
        return results
    
    def visualize_model_performance(self, results):
        """
        Create model performance visualizations
        """
        print("Creating model performance visualizations...")
        
        # Model comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        # Accuracy comparison
        bars1 = axes[0].bar(model_names, accuracies, color='skyblue', alpha=0.8)
        axes[0].set_title('Test Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(0.8, 1.0)
        
        for bar, acc in zip(bars1, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Cross-validation scores
        axes[1].bar(model_names, cv_means, yerr=cv_stds, capsize=5, 
                   color='lightcoral', alpha=0.8)
        axes[1].set_title('Cross-Validation Scores')
        axes[1].set_ylabel('CV Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Confusion matrix for best model
        cm = confusion_matrix(self.y_test, results[self.best_model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[2])
        axes[2].set_title(f'Confusion Matrix\n{self.best_model_name}')
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed performance summary
        print("\nDETAILED MODEL PERFORMANCE:")
        print("=" * 70)
        for name, result in results.items():
            marker = "🏆" if name == self.best_model_name else "  "
            print(f"{marker} {name:18} | Acc: {result['accuracy']:.4f} | "
                  f"CV: {result['cv_mean']:.4f}±{result['cv_std']:.4f}")
        
        print("Model performance visualizations completed!")
    
    def predict_message(self, message):
        """
        Predict if a message is spam or ham
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Create temporary dataframe for processing
        temp_df = pd.DataFrame({'message': [message]})
        temp_df = self.extract_features(temp_df)
        temp_df['cleaned_message'] = temp_df['message'].apply(self.preprocess_text)
        
        # Prepare features
        X_text = self.vectorizer.transform(temp_df['cleaned_message'])
        
        numerical_features = ['char_count', 'word_count', 'avg_word_length', 
                            'exclamation_count', 'question_count', 'capital_count',
                            'digit_count', 'financial_words', 'urgency_words', 
                            'action_words', 'phone_numbers', 'urls']
        
        X_numerical = temp_df[numerical_features].values
        X_combined = hstack([X_text, X_numerical])
        
        # Make prediction
        prediction = self.best_model.predict(X_combined)[0]
        if hasattr(self.best_model, 'predict_proba'):
            prediction_proba = self.best_model.predict_proba(X_combined)[0]
        else:
            prediction_proba = [0.5, 0.5]  # Default probabilities
        
        # Format results
        label = "SPAM" if prediction == 1 else "HAM"
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        return {
            'prediction': label,
            'confidence': confidence,
            'spam_probability': prediction_proba[1],
            'features': {
                'length': temp_df['char_count'].iloc[0],
                'words': temp_df['word_count'].iloc[0],
                'financial_words': temp_df['financial_words'].iloc[0],
                'urgency_words': temp_df['urgency_words'].iloc[0],
                'exclamations': temp_df['exclamation_count'].iloc[0]
            }
        }

def main():
    """
    Main function to run the SMS spam detection system
    """
    print("SMS SPAM DETECTION SYSTEM")
    print("=" * 60)
    
    # Initialize detector
    detector = SimpleSMSSpamDetector()
    
    # Load data
    df = detector.load_data("spam (1).csv")
    if df is None:
        print("Failed to load dataset. Exiting...")
        return
    
    # Extract features
    df = detector.extract_features(df)
    
    # Visualize dataset after preprocessing
    detector.visualize_dataset(df)
    
    # Analyze data
    detector.analyze_data(df)
    
    # Train models
    results = detector.train_models(df)
    
    # Visualize model performance after training
    detector.visualize_model_performance(results)
    
    # Test with sample messages
    test_messages = [
        "Congratulations! You've won $1000! Call 123-456-7890 now to claim!",
        "Hey, are you free for lunch tomorrow?",
        "URGENT: Your account will be suspended! Click here immediately!",
        "Thanks for the birthday wishes! Had a great time.",
        "FREE entry! Win £500 cash prize! Text WIN to 12345!",
        "Can you pick up milk on your way home?",
        "Limited time offer! 90% discount! Buy now!"
    ]
    
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE MESSAGES")
    print("="*60)
    
    for i, message in enumerate(test_messages, 1):
        result = detector.predict_message(message)
        
        print(f"\nMessage {i}: {message}")
        print(f"   Prediction: {result['prediction']} (Confidence: {result['confidence']:.1%})")
        print(f"   Spam Probability: {result['spam_probability']:.1%}")
        
        # Show key features
        features = result['features']
        key_features = [f"{k}: {v}" for k, v in features.items() if v > 0]
        if key_features:
            print(f"   Key features: {', '.join(key_features)}")
    
    print("\nAnalysis completed successfully!")
    print("Model trained and ready for predictions!")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Simple SMS Spam Detection System
Works with basic Python libraries
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

class SimpleSMSSpamDetector:
    """
    Simple SMS Spam Detection System
    """

    def __init__(self):
        """Initialize the detector"""
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                          'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                          'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do',
                          'does', 'did', 'will', 'would', 'could', 'should', 'i', 'you',
                          'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}

        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )

        self.best_model = None
        self.best_model_name = ""

    def load_data(self, file_path="spam (1).csv"):
        """
        Load the SMS dataset
        """
        print("Loading SMS dataset...")

        try:
            # Try different encodings
            for encoding in ['latin1', 'utf-8', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read file with any encoding")

            print(f"Dataset loaded with {encoding} encoding")

            # Clean the dataset
            df = df.iloc[:, :2]  # Keep only first two columns
            df.columns = ['label', 'message']
            df = df.dropna()  # Remove missing values

            # Convert labels to binary
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})

            print(f"Dataset Statistics:")
            print(f"   Total messages: {len(df):,}")
            print(f"   Ham messages: {sum(df['label'] == 0):,}")
            print(f"   Spam messages: {sum(df['label'] == 1):,}")
            print(f"   Spam percentage: {(sum(df['label'] == 1)/len(df)*100):.1f}%")

            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_text(self, text):
        """
        Basic text preprocessing
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)

        # Remove phone numbers
        text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', '', text)

        # Remove extra whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]

        return ' '.join(words)

    def extract_features(self, df):
        """
        Extract comprehensive features from messages
        """
        print("Extracting features...")

        # Basic text features
        df['char_count'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()
        df['avg_word_length'] = df['message'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )

        # Punctuation features
        df['exclamation_count'] = df['message'].str.count('!')
        df['question_count'] = df['message'].str.count(r'\?')
        df['capital_count'] = df['message'].str.count(r'[A-Z]')
        df['digit_count'] = df['message'].str.count(r'\d')

        # Spam-specific features
        currency_pattern = r'[$£€¥₹]|\b(money|cash|prize|win|free|offer|discount)\b'
        df['financial_words'] = df['message'].str.count(currency_pattern, flags=re.IGNORECASE)

        urgency_pattern = r'\b(urgent|asap|immediately|now|hurry|limited|expire|act)\b'
        df['urgency_words'] = df['message'].str.count(urgency_pattern, flags=re.IGNORECASE)

        action_pattern = r'\b(click|call|text|reply|send|subscribe|buy|order)\b'
        df['action_words'] = df['message'].str.count(action_pattern, flags=re.IGNORECASE)

        # Contact features
        df['phone_numbers'] = df['message'].str.count(r'\b\d{5,}\b')
        df['urls'] = df['message'].str.count(r'http|www|\.com|\.org|\.net')

        print("Feature extraction completed!")
        return df

    def visualize_dataset(self, df):
        """
        Create comprehensive dataset visualizations
        """
        print("Creating dataset visualizations...")

        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SMS Spam Detection - Dataset Analysis', fontsize=16, fontweight='bold')

        # 1. Label distribution
        labels = ['Ham', 'Spam']
        counts = df['label'].value_counts().sort_index()
        colors = ['#2ecc71', '#e74c3c']

        axes[0, 0].bar(labels, counts, color=colors, alpha=0.8)
        axes[0, 0].set_title('Message Distribution')
        axes[0, 0].set_ylabel('Count')
        for i, count in enumerate(counts):
            axes[0, 0].text(i, count + 50, str(count), ha='center', fontweight='bold')

        # 2. Message length distribution
        df.boxplot(column='char_count', by='label', ax=axes[0, 1])
        axes[0, 1].set_title('Message Length by Type')
        axes[0, 1].set_xlabel('Label (0=Ham, 1=Spam)')
        axes[0, 1].set_ylabel('Character Count')

        # 3. Word count distribution
        df.boxplot(column='word_count', by='label', ax=axes[0, 2])
        axes[0, 2].set_title('Word Count by Type')
        axes[0, 2].set_xlabel('Label (0=Ham, 1=Spam)')
        axes[0, 2].set_ylabel('Word Count')

        # 4. Financial words
        spam_financial = df[df['label'] == 1]['financial_words'].mean()
        ham_financial = df[df['label'] == 0]['financial_words'].mean()
        axes[1, 0].bar(['Ham', 'Spam'], [ham_financial, spam_financial],
                      color=colors, alpha=0.8)
        axes[1, 0].set_title('Avg Financial Words')
        axes[1, 0].set_ylabel('Average Count')

        # 5. Urgency words
        spam_urgency = df[df['label'] == 1]['urgency_words'].mean()
        ham_urgency = df[df['label'] == 0]['urgency_words'].mean()
        axes[1, 1].bar(['Ham', 'Spam'], [ham_urgency, spam_urgency],
                      color=colors, alpha=0.8)
        axes[1, 1].set_title('Avg Urgency Words')
        axes[1, 1].set_ylabel('Average Count')

        # 6. Exclamation marks
        spam_excl = df[df['label'] == 1]['exclamation_count'].mean()
        ham_excl = df[df['label'] == 0]['exclamation_count'].mean()
        axes[1, 2].bar(['Ham', 'Spam'], [ham_excl, spam_excl],
                      color=colors, alpha=0.8)
        axes[1, 2].set_title('Avg Exclamation Marks')
        axes[1, 2].set_ylabel('Average Count')

        plt.tight_layout()
        plt.show()

        # Feature correlation heatmap
        plt.figure(figsize=(12, 8))
        feature_cols = ['char_count', 'word_count', 'exclamation_count', 'capital_count',
                       'financial_words', 'urgency_words', 'action_words', 'phone_numbers']
        correlation_matrix = df[feature_cols + ['label']].corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print("Dataset visualizations completed!")
        return df

    def analyze_data(self, df):
        """
        Basic data analysis
        """
        print("\n" + "="*60)
        print("DATA ANALYSIS RESULTS")
        print("="*60)

        print(f"\nMessage Length Analysis:")
        print(f"Average Ham length: {df[df['label']==0]['char_count'].mean():.1f} characters")
        print(f"Average Spam length: {df[df['label']==1]['char_count'].mean():.1f} characters")

        print(f"\nWord Count Analysis:")
        print(f"Average Ham words: {df[df['label']==0]['word_count'].mean():.1f}")
        print(f"Average Spam words: {df[df['label']==1]['word_count'].mean():.1f}")

        print(f"\nSpam Indicators:")
        print(f"Financial words - Ham avg: {df[df['label']==0]['financial_words'].mean():.2f}, Spam avg: {df[df['label']==1]['financial_words'].mean():.2f}")
        print(f"Urgency words - Ham avg: {df[df['label']==0]['urgency_words'].mean():.2f}, Spam avg: {df[df['label']==1]['urgency_words'].mean():.2f}")
        print(f"Exclamations - Ham avg: {df[df['label']==0]['exclamation_count'].mean():.2f}, Spam avg: {df[df['label']==1]['exclamation_count'].mean():.2f}")

        # Show some examples
        print(f"\nSample SPAM messages:")
        spam_samples = df[df['label']==1]['message'].head(3)
        for i, msg in enumerate(spam_samples, 1):
            print(f"  {i}. {msg}")

        print(f"\nSample HAM messages:")
        ham_samples = df[df['label']==0]['message'].head(3)
        for i, msg in enumerate(ham_samples, 1):
            print(f"  {i}. {msg}")

    def train_models(self, df):
        """
        Train multiple machine learning models
        """
        print("\nTraining machine learning models...")

        # Prepare text features
        df['cleaned_message'] = df['message'].apply(self.preprocess_text)
        X_text = self.vectorizer.fit_transform(df['cleaned_message'])

        # Prepare numerical features
        numerical_features = ['char_count', 'word_count', 'avg_word_length',
                            'exclamation_count', 'question_count', 'capital_count',
                            'digit_count', 'financial_words', 'urgency_words',
                            'action_words', 'phone_numbers', 'urls']

        X_numerical = df[numerical_features].values

        # Combine features
        X_combined = hstack([X_text, X_numerical])
        y = df['label'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )

        # Store for later use
        self.X_test, self.y_test = X_test, y_test

        # Define models
        models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }

        # Train and evaluate models
        results = {}
        best_score = 0

        print("\n" + "="*60)
        print("MODEL TRAINING RESULTS")
        print("="*60)

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }

            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Cross-Validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Track best model
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name

        print(f"\nBest model: {self.best_model_name} (CV: {best_score:.4f})")

        # Detailed report for best model
        print(f"\nDETAILED CLASSIFICATION REPORT - {self.best_model_name}:")
        print("-" * 60)
        best_pred = results[self.best_model_name]['predictions']
        print(classification_report(y_test, best_pred, target_names=['Ham', 'Spam']))

        return results

    def visualize_model_performance(self, results):
        """
        Create model performance visualizations
        """
        print("Creating model performance visualizations...")

        # Model comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]

        # Accuracy comparison
        bars1 = axes[0].bar(model_names, accuracies, color='skyblue', alpha=0.8)
        axes[0].set_title('Test Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(0.8, 1.0)

        for bar, acc in zip(bars1, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # Cross-validation scores
        axes[1].bar(model_names, cv_means, yerr=cv_stds, capsize=5,
                   color='lightcoral', alpha=0.8)
        axes[1].set_title('Cross-Validation Scores')
        axes[1].set_ylabel('CV Score')
        axes[1].tick_params(axis='x', rotation=45)

        # Confusion matrix for best model
        cm = confusion_matrix(self.y_test, results[self.best_model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[2])
        axes[2].set_title(f'Confusion Matrix\n{self.best_model_name}')
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')

        plt.tight_layout()
        plt.show()

        # Print detailed performance summary
        print("\nDETAILED MODEL PERFORMANCE:")
        print("=" * 70)
        for name, result in results.items():
            marker = "🏆" if name == self.best_model_name else "  "
            print(f"{marker} {name:18} | Acc: {result['accuracy']:.4f} | "
                  f"CV: {result['cv_mean']:.4f}±{result['cv_std']:.4f}")

        print("Model performance visualizations completed!")

    def predict_message(self, message):
        """
        Predict if a message is spam or ham
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet!")

        # Create temporary dataframe for processing
        temp_df = pd.DataFrame({'message': [message]})
        temp_df = self.extract_features(temp_df)
        temp_df['cleaned_message'] = temp_df['message'].apply(self.preprocess_text)

        # Prepare features
        X_text = self.vectorizer.transform(temp_df['cleaned_message'])

        numerical_features = ['char_count', 'word_count', 'avg_word_length',
                            'exclamation_count', 'question_count', 'capital_count',
                            'digit_count', 'financial_words', 'urgency_words',
                            'action_words', 'phone_numbers', 'urls']

        X_numerical = temp_df[numerical_features].values
        X_combined = hstack([X_text, X_numerical])

        # Make prediction
        prediction = self.best_model.predict(X_combined)[0]
        if hasattr(self.best_model, 'predict_proba'):
            prediction_proba = self.best_model.predict_proba(X_combined)[0]
        else:
            prediction_proba = [0.5, 0.5]  # Default probabilities

        # Format results
        label = "SPAM" if prediction == 1 else "HAM"
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]

        return {
            'prediction': label,
            'confidence': confidence,
            'spam_probability': prediction_proba[1],
            'features': {
                'length': temp_df['char_count'].iloc[0],
                'words': temp_df['word_count'].iloc[0],
                'financial_words': temp_df['financial_words'].iloc[0],
                'urgency_words': temp_df['urgency_words'].iloc[0],
                'exclamations': temp_df['exclamation_count'].iloc[0]
            }
        }

# Initialize detector
detector = SimpleSMSSpamDetector()

# Load data
df = detector.load_data("spam (1).csv")
if df is None:
    print("Failed to load dataset. Exiting...")
#     return # Commenting this out to allow the rest of the script to run if data loading fails

# Extract features
if df is not None:
    df = detector.extract_features(df)

    # Visualize dataset after preprocessing
    detector.visualize_dataset(df)

    # Analyze data
    detector.analyze_data(df)

    # Train models
    results = detector.train_models(df)

    # Visualize model performance after training
    detector.visualize_model_performance(results)

    # Test with sample messages
    test_messages = [
        "Congratulations! You've won $1000! Call 123-456-7890 now to claim!",
        "Hey, are you free for lunch tomorrow?",
        "URGENT: Your account will be suspended! Click here immediately!",
        "Thanks for the birthday wishes! Had a great time.",
        "FREE entry! Win £500 cash prize! Text WIN to 12345!",
        "Can you pick up milk on your way home?",
        "Limited time offer! 90% discount! Buy now!"
    ]

    print("\n" + "="*60)
    print("TESTING WITH SAMPLE MESSAGES")
    print("="*60)

    for i, message in enumerate(test_messages, 1):
        result = detector.predict_message(message)

        print(f"\nMessage {i}: {message}")
        print(f"   Prediction: {result['prediction']} (Confidence: {result['confidence']:.1%})")
        print(f"   Spam Probability: {result['spam_probability']:.1%}")

        # Show key features
        features = result['features']
        key_features = [f"{k}: {v}" for k, v in features.items() if v > 0]
        if key_features:
            print(f"   Key features: {', '.join(key_features)}")

    print("\nAnalysis completed successfully!")
    print("Model trained and ready for predictions!")

# if __name__ == "__main__":
#     main() # Commenting this out to allow the detector object to be available globally