import json
import re
import nltk
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score, log_loss
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# from transformers import AdamW

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    '''
    def __getitem__(self, idx):
        if idx >= len(self.encodings['input_ids']):  # assuming 'input_ids' is a key in encodings
            raise IndexError(
                f"Index {idx} is out of bounds for this dataset with length {len(self.encodings['input_ids'])}")
        input_ids = torch.tensor(self.encodings['input_ids'][idx]).unsqueeze(0)  # Add a batch dimension
        attention_mask = torch.tensor(self.encodings['attention_mask'][idx]).unsqueeze(0)  # Add a batch dimension
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item
    '''

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))  # russian stop words


def clean_text(text):  # removing the links, transforming to lower register, removing the symbols
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', ' ', text)
    # text = re.sub(r'\s+', ' ', text).strip() - only if we wanna remove whole symbols
    return text.lower()


def remove_stop_words(text_list):  # removing some russian stop words
    cleaned_text_list = []
    for text in text_list:
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        cleaned_text_list.append(' '.join(filtered_words))
    return cleaned_text_list


def add_context_embedding(texts, contexts):
    new_texts = []
    for i, text in enumerate(texts):
        context = contexts.get(i)  # return None if the key is not found
        if context is not None:
            new_text = context['before'] + text + context['after']
            new_texts.append(new_text)
        else:
            new_texts.append(text)  # handle the case where context is not found
    return new_texts


def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        texts = []
        labels = []
        label_dict = {'positive': 0, 'neutral': 1, 'negative': 2}
        for item in data:
            if 'text' in item and 'sentiment' in item:
                if item['sentiment'] in label_dict:
                    texts.append(item['text'])
                    labels.append(label_dict[item['sentiment']])
        return texts, labels
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main():
    torch.backends.cudnn.benchmark = True

    texts, labels = load_data('train.json')
    if texts is None or labels is None:
        return

    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(texts)):
        print(f"\nFold {fold+1}/4")
        
        # Split data
        train_texts = [texts[i] for i in train_index]
        val_texts = [texts[i] for i in val_index]
        train_labels = [labels[i] for i in train_index]
        val_labels = [labels[i] for i in val_index]

        # Calculate class weights
        unique_classes = np.unique(train_labels)
        weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=train_labels)
        weights_tensor = torch.tensor(weights, dtype=torch.float)

        # Text preprocessing first
        cleaned_train_texts = [clean_text(text) for text in train_texts]
        cleaned_val_texts = [clean_text(text) for text in val_texts]
        
        processed_train_texts = remove_stop_words(cleaned_train_texts)
        processed_val_texts = remove_stop_words(cleaned_val_texts)

        # Apply SMOTE after preprocessing
        vectorizer = TfidfVectorizer()
        X_train_vect = vectorizer.fit_transform(processed_train_texts)
        sm = SMOTE(random_state=42)
        X_train_vect_resampled, train_labels_resampled = sm.fit_resample(X_train_vect, train_labels)
        
        # Convert back to texts without inverse_transform
        train_texts_resampled = vectorizer.inverse_transform(X_train_vect_resampled)
        train_texts_resampled = [' '.join(text) for text in train_texts_resampled]  # Convert to strings
        train_labels = train_labels_resampled

        # Add context
        additional_context = {'before': 'Previous sentence.', 'after': 'Next sentence.'}
        train_texts_with_context = add_context_embedding(train_texts_resampled, 
                                                       {i: additional_context for i in range(len(train_texts_resampled))})
        val_texts_with_context = add_context_embedding(processed_val_texts, 
                                                     {i: additional_context for i in range(len(processed_val_texts))})

        # Initialize model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-multilingual-cased',
            num_labels=3
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Tokenize
        train_encodings = tokenizer.batch_encode_plus(
            train_texts_with_context,
            max_length=32,
            padding='longest',
            truncation=True
        )
        val_encodings = tokenizer.batch_encode_plus(
            val_texts_with_context,
            max_length=32,
            padding='longest',
            truncation=True
        )

        # Create datasets and dataloaders
        train_dataset = SentimentDataset(train_encodings, train_labels)
        val_dataset = SentimentDataset(val_encodings, val_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=16,  # Increased from 4 to 16
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2  # Add prefetching
        )
        val_loader = DataLoader(val_dataset, batch_size=4)

        # Initialize optimizer and scaler
        optimizer = AdamW(model.parameters(), lr=9e-5, eps=1e-6)
        scaler = GradScaler()

        # Training loop
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_accuracy = 0.0

        # Add lists to track metrics
        train_losses = []
        val_losses = []
        
        print(f"Training samples: {len(train_texts_resampled)}")
        print(f"Validation samples: {len(val_texts)}")
        
        accumulation_steps = 4  # Accumulate gradients over 4 batches

        for epoch in range(5):
            print(f"\nEpoch {epoch+1}/5")
            
            # Training
            model.train()
            total_loss = 0
            batch_count = len(train_loader)
            
            print("Training:")
            for batch_idx, batch in enumerate(train_loader, 1):
                if batch_idx % 10 == 0:  # Print progress every 10 batches
                    print(f"Batch {batch_idx}/{batch_count} "
                          f"({(batch_idx/batch_count)*100:.1f}%)", end='\r')
                
                optimizer.zero_grad()
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss / accumulation_steps  # Scale loss

                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1}, Average training loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            total_val_loss = 0
            predictions = []
            true_labels = []
            probabilities = []
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels'].to(device)
                    outputs = model(**inputs, labels=labels)
                    
                    total_val_loss += outputs.loss.item()
                    probs = torch.softmax(outputs.logits, dim=1)
                    preds = torch.argmax(outputs.logits, dim=1)
                    
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())

            # Calculate metrics
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')
            bal_acc = balanced_accuracy_score(true_labels, predictions)
            
            # Calculate confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            
            # Calculate log loss
            try:
                log_loss_value = log_loss(true_labels, probabilities)
            except ValueError:
                log_loss_value = float('nan')

            print(f"Validation loss: {avg_val_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Balanced Accuracy: {bal_acc:.4f}")
            print(f"Log Loss: {log_loss_value:.4f}")

            # Plot confusion matrix for each epoch
            plot_confusion_matrix(cm, f'confusion_matrix_fold_{fold}_epoch_{epoch+1}.png')

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'balanced_accuracy': bal_acc,
                        'confusion_matrix': cm,
                    }, f'best_model_fold_{fold}.pt')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= 3:
                    print("Early stopping triggered")
                    break

            # Plot training history at the end of each epoch
            plot_training_history(train_losses, val_losses, f'training_history_fold_{fold}.png')

        # Store fold results with additional metrics
        fold_results.append({
            'fold': fold,
            'best_accuracy': best_accuracy,
            'best_val_loss': best_val_loss,
            'f1_score': f1,
            'balanced_accuracy': bal_acc,
            'log_loss': log_loss_value
        })

    # Print final results
    print("\nFinal Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}:")
        print(f"  Accuracy = {result['best_accuracy']:.4f}")
        print(f"  F1 = {result['f1_score']:.4f}")
        print(f"  Balanced Accuracy = {result['balanced_accuracy']:.4f}")
        print(f"  Log Loss = {result['log_loss']:.4f}")

if __name__ == '__main__':
    main()

