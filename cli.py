import torch

# Define your model architecture to match the trained model
class LSTMModel(torch.nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=11, hidden_size=100, num_layers=1, batch_first=True)  # Adjust hidden_size
        self.fc1 = torch.nn.Linear(100, 50)  # Example layer
        self.fc2 = torch.nn.Linear(50, 25)   # Example layer
        self.fc3 = torch.nn.Linear(25, 1)    # Output layer for a single value
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Get LSTM output
        x = lstm_out[:, -1, :]  # Use the last time step's output
        x = torch.relu(self.fc1(x))  # Pass through first layer
        x = torch.relu(self.fc2(x))  # Pass through second layer
        output = self.fc3(x)  # Final output layer
        return output  # Return a single value

# Load the entire model
model = torch.load('model.pt')  # Adjust to 'cpu' if needed
model.eval()  # Set the model to evaluation mode

# Function to predict output
def predict(input_values):
    # Convert input values to tensor
    input_tensor = torch.tensor(input_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_tensor)
    return output.item()  # Return a single value

# Get comma-separated input from CLI
input_str = input("Enter 11 comma-separated values: ")
input_values = list(map(float, input_str.split(',')))

if len(input_values) != 11:
    print("Please enter exactly 11 values.")
else:
    prediction = predict(input_values)
    print("Predicted output:", prediction)