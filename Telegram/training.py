import random 
import json 
import pickle 
import numpy as np 
import nltk 
from keras.models import Sequential 
from nltk.stem import WordNetLemmatizer 
from keras.layers import Dense, Activation, Dropout 
from keras.optimizers import SGD 

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer() 

# Read the intents JSON file
intents = json.loads(open("intense.json").read())

# Initialize empty lists to store data
words = [] 
classes = [] 
documents = [] 
ignore_letters = ["?", "!", ".", ","] 

# Extract words, classes, and documents from intents
for intent in intents['intents']: 
    for pattern in intent['patterns']: 
        # Tokenize words from patterns
        word_list = nltk.word_tokenize(pattern) 
        words.extend(word_list) # Add words to the words list 
        
        # Associate patterns with respective tags 
        documents.append((word_list, intent['tag'])) 

        # Append tags to the classes list 
        if intent['tag'] not in classes: 
            classes.append(intent['tag']) 

# Lemmatize words and remove ignore letters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters] 
words = sorted(set(words)) 

# Save words and classes lists to binary files 
pickle.dump(words, open('words.pkl', 'wb')) 
pickle.dump(classes, open('classes.pkl', 'wb')) 

# Create training data
training = [] 
output_empty = [0] * len(classes) 

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create a bag of words with fixed size
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Make a copy of the output_empty
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])


# Shuffle the training data
random.shuffle(training) 


# Split the data into input and output
train_x = [row[0] for row in training]
train_y = [row[1] for row in training]

# Convert lists to numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

# Create a Sequential model
model = Sequential() 
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(len(train_y[0]), activation='softmax')) 

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=500, batch_size=5, verbose=1) 

# Save the model
model.save("chatbotmodel.h5")
