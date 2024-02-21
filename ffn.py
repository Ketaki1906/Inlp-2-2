import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Build model
class MyEntityPredictor(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, p: int, s: int):
        super().__init__()
        self.embedding_module = torch.nn.Embedding(vocabulary_size, embedding_dim)
        input_dim =(p + s + 1) * embedding_dim 
        self.entity_predictor = torch.nn.Sequential(
                                        torch.nn.Linear(input_dim,hidden_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_dim, output_dim))
    
    def forward(self, word_indices: torch.Tensor):
        # Take the word indices and return an "unnormalized probability distribution" over the labels (1 and 0)
        embeddings = self.embedding_module(word_indices)
        # Reshape the embeddings to flatten them into a single dimension
        embeddings_flat = embeddings.view(embeddings.size(0), -1)
        return self.entity_predictor(embeddings_flat)


def load_glove_embeddings(glove_path):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]])
            embeddings[word] = vector

    return embeddings,len(embeddings[word])

def create_vocabulary(data,p,s):
        vocabulary = {}
        pos_vocabulary={}
        pos_word_dict={}

        vocabulary['<s>']=len(vocabulary)+1
        vocabulary['</s>']=len(vocabulary)+1  
        vocabulary['<ukn>']=len(vocabulary)+1  
        pos_vocabulary['START_TAG']=0
        pos_vocabulary['END_TAG']=1   
        pos_vocabulary['UKN']=2  
        pos_word_dict['<s>']='START_TAG'  
        pos_word_dict['</s>']='END_TAG'    
        pos_word_dict['<ukn>']='UKN'  
        for line in data.split('\n'):
           
            if line.strip() and not line.startswith('#'):
                columns = line.split('\t')
                word = columns[1] 
                pos_tag=columns[3] 
                pos_word_dict[word]=pos_tag
                vocabulary[word] = vocabulary.get(word, len(vocabulary) + 1)  
                pos_vocabulary[pos_tag] = pos_vocabulary.get(pos_tag, len(pos_vocabulary) + 1)           

      
        return vocabulary,pos_vocabulary,pos_word_dict

def create_x_y_train(vocabulary,pos_vocabulary,p,s,data,pos_word_dict):
        x_train=[]
        y_train=[]
        count=0
        for line in data.split('\n'):
            if line.strip() and line.startswith('# text'):
                    count+=1
                    columns_obtained = line.split(' ')
                    columns_obtained[2]='<s>'
                    columns=columns_obtained[2:]
                    columns.append('</s>')
                    
                    for i in range(len(columns)+1):
                        if i+p+s<len(columns):
                            window_x=[]
                            window_y=[]
                            for word in columns[i:i+p+s+1]:
                                window_x.append(vocabulary[word])
                                window_y.append(pos_vocabulary[pos_word_dict[word]])
                            x_train.append(window_x)
                            y_train.append(window_y)
                        else:
                            break
                    # break   
            if count==4232:
                    break
        # Convert lists to PyTorch tensors
        x_train_tensor = torch.tensor(x_train)
        y_train_tensor = torch.tensor(y_train)
        return x_train_tensor,y_train_tensor


def create_x_y_test(vocabulary,pos_vocabulary,p,s,data,pos_word_dict):
        x_test=[]
        y_test=[]
        count=0
        for line in data.split('\n'):

            if line.strip() and line.startswith('# text'):
                count=+1
                columns_obtained = line.split(' ')
                columns_obtained[2]='<s>'
                columns=columns_obtained[2:]
                columns.append('</s>')
                
                for i in range(len(columns)+1):
                    if i+p+s+1<len(columns)+1:
                        window_x=[]
                        window_y=[]
                        for word in columns[i:i+p+s+1]:
                            if vocabulary[word]:
                                window_x.append(vocabulary[word])
                                window_y.append(pos_vocabulary[pos_word_dict[word]])
                            else:
                                word='<ukn>'
                                window_x.append(vocabulary[word])
                                window_y.append(pos_vocabulary[pos_word_dict[word]])

                        x_test.append(window_x)
                        y_test.append(window_y)
                    else:
                        break
                

        # Convert lists to PyTorch tensors
        x_test_tensor = torch.tensor(x_test)
        y_test_tensor = torch.tensor(y_test)
        return x_test_tensor,y_test_tensor

def accuracy_fn(y_true, y_pred):
    # Assuming y_true and y_pred are both tensors
    correct_predictions = (y_pred == y_true).float()
    accuracy = correct_predictions.sum() / len(y_true)
    return accuracy.item()  # Return the accuracy as a Python float

def main():

    file_path = 'en_atis-ud-train.conllu' 

    with open(file_path, 'r') as file:
        data = file.read()

    # Create device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p = 0
    s = 0

    vocabulary,pos_vocab,pos_word_dict = create_vocabulary(data,p,s)

    x_train,y_train=create_x_y_train(vocabulary,pos_vocab,p,s,data,pos_word_dict)
    # print(x_train.size())
    x_test,y_test=create_x_y_test(vocabulary,pos_vocab,p,s,data,pos_word_dict)

    x_train_tensor, y_train_tensor = x_train.to(device), y_train.to(device)
    x_test_tensor, y_test_tensor = x_test.to(device), y_test.to(device)

    vocabulary_size = len(vocabulary)
    embedding_dim = 50
    hidden_dim = 1000  
    output_dim = len(pos_vocab)  
    model = MyEntityPredictor(vocabulary_size, embedding_dim, hidden_dim, output_dim, p, s).to(device)

    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    batch_size = 100
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 50

    for epoch in range(num_epochs):
        # Loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        model.train()  # Set the model to training mode
        for batch_x, batch_y in data_loader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, torch.max(batch_y, 1)[1])  # Assuming batch_y is one-hot encoded, use max to get indices
            loss.backward()
            optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test_tensor)
            test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = criterion(test_logits, torch.max(y_test_tensor, 1)[1])

            test_acc = accuracy_fn(y_true=y_test_tensor,y_pred=test_pred)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

if __name__ == "__main__":
    main()


