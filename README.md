# **DeepCaption: An AI-Powered Image Caption Generator**

This project tackles the challenging task of **image captioning** by combining a **ResNet-50 feature extractor** with an **LSTM-based sequence processor**. It is designed to generate accurate and descriptive text for a given image, bridging the gap between computer vision and natural language processing.

The work is a deep learning approach inspired by **Yumi's Blog**.

### **Real-World Applications**

The ability to automatically generate image captions has a wide range of practical applications:

* **Self-Driving Cars:** Captioning the environment around a vehicle can provide crucial context to its autonomous driving system.
* **Assistive Technology:** Converting images to speech can assist the visually impaired, helping them navigate their surroundings more independently.
* **Security & Surveillance:** Generating captions from CCTV footage could help in the automatic detection of suspicious or malicious activities, potentially reducing crime rates.
* **Enhanced Web Search:** By generating captions for images, search engines could provide more precise results based on the image's content.

***

### **Technical Breakdown**

#### **1. Data Preparation**
The model is trained using the **Flickr8K dataset**, a collection of 8,000 images, each with five distinct captions. The dataset is split into 6,000 images for training, 1,000 for validation, and 1,000 for testing. The raw data includes a file, `Flickr8k.token.txt`, which maps each image ID to its five captions.

#### **2. Text Preprocessing**
* **Cleaning**: The text is cleaned by converting all words to lowercase and removing punctuation, numbers, and special symbols. This reduces the vocabulary size and helps prevent overfitting. Stop words and stemming are deliberately not used, as they are crucial for generating natural and grammatically correct sentences.
* **Vocabulary**: A vocabulary is built from the cleaned text, and words that appear less than 10 times are removed to make the model more robust to outliers. The vocabulary is then augmented with `'startseq'` and `'endseq'` tokens to signal the beginning and end of each caption, respectively.

#### **3. Image Feature Extraction**
Instead of building a CNN from scratch, the project uses **transfer learning** with a pre-trained **ResNet-50 model**. The last classification layer of the ResNet-50 model is removed, allowing the model to act as a feature extractor. Each image is converted into a **2048-dimensional feature vector** which captures its essential visual information.

#### **4. Word Embeddings**
The captions are converted into numerical representations using **pre-trained GloVe word embeddings**. This eliminates the need to train a separate embedding layer from scratch and significantly reduces training time. The GloVe vectors (`glove.6B.50d.txt`) convert each word into a 50-dimensional vector.

#### **5. Data Generation**
The task is framed as a supervised learning problem where the model predicts the next word in a sequence given the image features and a partial caption. Due to the large size of the training data (over 3GB), a **custom Python generator function** is used to feed the model with data in batches. This approach prevents the entire dataset from being loaded into memory, making the training process much more efficient.

#### **6. Model Architecture**
The model is built using Keras's **Functional API** to handle the two distinct inputs (image features and text sequence). The architecture consists of:

* **Photo Feature Extractor**: A Dense layer that processes the 2048-element image vector into a 256-element representation.
* **Sequence Processor**: An Embedding layer (which is **frozen**) followed by a 256-unit **LSTM layer**.
* **Decoder**: The outputs of the two branches are merged and processed by a Dense layer with a **softmax activation function** to predict the probability distribution for the next word in the caption.

The model is trained using the **Adam optimizer** and **categorical cross-entropy loss**. A **30% dropout rate** is applied to reduce overfitting. The model learns to generate captions by predicting one word at a time, continuing until the `'endseq'` token is produced.

