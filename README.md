# 🏥 MedBot: GenAI Assistant for Medical NLP and Clinical Case Routing
Gen AI assistant to analyze **medical transcriptions and patient summaries**, using **LangGraph**, enabling dynamic routing, stateful memory, and agentic control flow for **clinical decision support**.

## 📘 Project Overview

This project demonstrates how **Generative AI techniques** can be combined to build a real-world assistant for healthcare support. The assistant, called **MedBot**, is designed to understand medical case descriptions, classify them into medical specialties, and offer relevant next steps, such as retrieving similar records or suggesting nearby providers.

The solution showcases both **model fine-tuning** and **tool-augmented conversation design** using **LangGraph** and **Gemini API**.

## Notebook
- 📓 Main notebook: [`capstone-project-medbot.ipynb`](./capstone-project-medbot.ipynb)
  
---

## 🎯 Objective

Build a domain-specific GenAI assistant that:

- Understands patient summaries or transcriptions
- Classifies the case into **Urology**, **Nephrology**, or **Other**
- Responds based on the category with context-aware suggestions


---

## 💡 Generative AI Techniques Used

| Technique                      | Purpose                                                                 |
|-------------------------------|-------------------------------------------------------------------------|
| **Prompt Engineering**         | Establish a baseline zero-shot classifier using Gemini                 |
| **Gemini API Fine-Tuning**     | Train a custom model on labeled medical transcriptions                 |
| **Semantic Embedding Evaluation** | Evaluate predictions via similarity to reference examples           |
| **LangGraph Tool Routing**     | Route between classifier, search, and user interaction nodes           |
| **Tool-Augmented Reasoning**   | Dynamically trigger tools like `classify_transcription` or `find_local_provider` |
| **Multi-Turn Chatbot with Memory** | Maintain and reason over evolving patient summaries               |

---

## 🧱 Project Structure

### **Phase 1: Fine-Tuning a Model for Medical Classification**

- **1.** Load dependencies
- **2.** Prepare and clean medical transcription dataset
- **3.** Prompt-based zero-shot classification (baseline)
- **4.** Evaluate predictions using embeddings
- **5.** Fine-tune Gemini model
- **6.** Compare and validate performance

### **Phase 2: Building a LangGraph Chatbot**

- **1.** Load and configure environment
- **2.** Define MedBot state and welcome logic
- **3.** Add human interaction and looping
- **4.** Integrate classification tool (tuned/baseline)
- **5.** Route classification responses and follow-up tools
- **6.** Add simulated ground search for provider lookup
- **7.** Plan retrieval tool for similar case search

---

## ✅ Capstone Alignment

This project satisfies core capstone objectives:

- ✅ Use of **LLM APIs** and prompt engineering
- ✅ Application of **fine-tuning and zero-shot comparison**
- ✅ Creation of an **interactive GenAI system** using LangGraph
- ✅ Clear **modular logic** with multi-tool orchestration
- ✅ Strong **domain-specific use case** in healthcare

---
---
# Notebook
---
---

## 1. Load dependencies


```python
!pip uninstall -qqy jupyterlab  # Remove unused packages from Kaggle's base image that conflict
!pip install -U -q "google-genai==1.7.0"
```


```python
from google import genai
from google.genai import types

genai.__version__
```

### Set up your API key

To run the following cell, your API key must be stored it in a [Kaggle secret](https://www.kaggle.com/discussions/product-feedback/114053) named `GOOGLE_API_KEY`.

If you don't already have an API key, you can grab one from [AI Studio](https://aistudio.google.com/app/apikey). You can find [detailed instructions in the docs](https://ai.google.dev/gemini-api/docs/api-key).

To make the key available through Kaggle secrets, choose `Secrets` from the `Add-ons` menu and follow the instructions to add your key or enable it for this notebook.


```python
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Explore available models

You will be using the [`TunedModel.create`](https://ai.google.dev/api/tuning#method:-tunedmodels.create) API method to start the fine-tuning job and create your custom model. Find a model that supports it through the [`models.list`](https://ai.google.dev/api/models#method:-models.list) endpoint. You can also find more information about tuning models in [the model tuning docs](https://ai.google.dev/gemini-api/docs/model-tuning/tutorial?lang=python).


```python
for model in client.models.list():
    if "createTunedModel" in model.supported_actions:
        print(model.name)
```

## 2. Dataset Preparation & Label Cleaning

### 📁 Dataset: Medical Transcriptions

The [**Medical Transcriptions**](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) dataset has been added to this notebook as an input under the **`/kaggle/input/medicaltranscriptions`** directory.

It contains a CSV file with transcribed medical records across various specialties and report types. This dataset can be used for tasks such as:

- **Text classification** (e.g., predicting medical specialty from transcription text)
- **Named Entity Recognition (NER)** for extracting patient symptoms, medications, or diagnoses
- **Fine-tuning language models** for medical domain understanding

#### 📄 File Structure
- `medicaltranscriptions.csv`: The primary dataset file containing the transcribed medical reports.

#### 🧾 Key Columns
- `Medical Specialty`: The category of the transcription (e.g., Cardiology, Radiology).
- `Sample Name`: A brief title or label for the transcription.
- `Transcription`: The full text of the medical report.

We will use this dataset to fine-tune a custom language model for classification tasks.



```python
import pandas as pd

df = pd.read_csv('/kaggle/input/medicaltranscriptions/mtsamples.csv', index_col=0)


# Get the full sorted list of unique sample_name values
medical_specialty_list = sorted(df['medical_specialty'].dropna().unique())

df.head()
```


```python
df = df.dropna(subset=['transcription'])

valid_specialties= [' Urology', ' Nephrology',]
# Step 3: Filter the DataFrame to keep only those rows
df = df[df['medical_specialty'].isin(valid_specialties)]

print(df.shape)

df['medical_specialty'].value_counts()
```


```python

import re

def preprocess_transcription(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    return text.strip()[:40000]



df['textInput'] = df['transcription'].apply(preprocess_transcription)
df['output'] = df['medical_specialty']
df_tune = df[['textInput', 'output']]
tune_data = {'examples': df_tune.to_dict(orient='records')}
```


```python
def sample_data(df, num_samples):
    return (
        df.groupby("output", group_keys=False)
          .apply(lambda x: x.sample(min(len(x), num_samples), random_state=42))
          .reset_index(drop=True)
    )



TRAIN_NUM_SAMPLES = 50
TEST_NUM_SAMPLES = 20

df_train = sample_data(df, TRAIN_NUM_SAMPLES)
df_test = sample_data(df, TEST_NUM_SAMPLES)

```


```python
sample_idx = 0
sample_row = df_test.iloc[sample_idx]['textInput']
sample_label = df_test.iloc[sample_idx]['output']

print(sample_row)
print('---')
print('Actual Label:', sample_label)

```

## 3. Prompt Engineering Baseline Model
In this step, we evaluated the capabilities of Gemini models for medical transcription classification using prompt engineering. The goal was to classify each transcription into a medical specialty (e.g., Urology or Nephrology) without any fine-tuning.


### 3.1 Zero-shot Prompt Engineering
We first tested Gemini using a **naïve zero-shot prompt**, directly asking the model what category a transcription belongs to:


```python
# Ask the model directly in a zero-shot prompt.
prompt = "What category does the following medical transcription belong to?"

response = client.models.generate_content(
    model="gemini-1.5-flash-001",
    contents=[prompt, sample_row]
)

print(response.text)

```

This initial approach helped assess whether the model could infer medical context from raw clinical notes without prior task definition. While it sometimes returned correct responses, the output lacked consistency and structure.

### 3.2 Prompt-Engineered Function (with System Instruction)
To improve reliability, we crafted a structured prompt using Gemini's system_instruction feature. This clarified the model’s role as a classification service. We then wrapped this in a callable prediction function.


```python
from google.api_core import retry
from google.genai import types

# Define system instruction for classification
system_instruct = """
You are a classification service. You will be passed input that represents
a medical transcription,and you must respond with the category it belongs to.
"""

# Retry handler for rate limits or service unavailability
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

@retry.Retry(predicate=is_retriable)
def predict_label(text: str) -> str:
    response = client.models.generate_content(
        model="gemini-1.5-flash-001",
        config=types.GenerateContentConfig(
            system_instruction=system_instruct),
        contents=text)

    rc = response.candidates[0]
    if rc.finish_reason.name != "STOP":
        return "(error)"
    else:
        return rc.content.parts[0].text.strip()



prediction = predict_label(sample_row)

print("Prediction:", prediction)
print("Actual:", sample_label)

```


```python
import tqdm
from tqdm.rich import tqdm as tqdmr
import warnings

tqdmr.pandas()  # 🔧 Activate tqdm for Pandas
warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)


# Re-sample 2 examples per class from df_test
def sample_data(df, num_samples):
    return (
        df.groupby("output", group_keys=False)
          .apply(lambda x: x.sample(min(len(x), num_samples), random_state=42))
          .reset_index(drop=True)
    )

df_baseline_eval = sample_data(df_test, 10)


## predict ocross the test set
df_baseline_eval['Prediction'] = df_baseline_eval['textInput'].progress_apply(predict_label)

```

## 4. Embedding-Based Evaluation

### 4.1 Define Embedding Function (Using Gemini)
What we’re doing:


We will define a retryable function that uses text-embedding-004 to generate an embedding vector for a given string. This is used to semantically compare category names.


```python
from google.api_core import retry
from google.genai import types

# Retry handler for rate limits or temporary issues
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

@retry.Retry(predicate=is_retriable, timeout=300.0)
def embed_fn(text: str) -> list[float]:
    """Get Gemini embedding for text (for classification)."""
    response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="classification",  # We're using this for label similarity
        ),
    )
    return response.embeddings[0].values

```

### 4.2 Create Embeddings for Predicted and Actual Labels
📌What we’re doing:
Apply embed_fn() to df_baseline_eval["output"] and ["Prediction"].

Store embeddings in new columns for later comparison.


```python
import numpy as np
import tqdm
from tqdm.rich import tqdm as tqdmr
import warnings

# Enable tqdm on pandas
tqdmr.pandas()
warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)

# Generate embeddings
df_baseline_eval["Actual_Embed"] = df_baseline_eval["output"].progress_apply(embed_fn)
df_baseline_eval["Predicted_Embed"] = df_baseline_eval["Prediction"].progress_apply(embed_fn)

```

 ### 4.3 Calculate Cosine Similarity
📌 What we're doing:
We’ll compute the cosine similarity between the predicted and actual label embeddings for each row. This gives us a numerical score (0–1) for how semantically close the predicted label is to the ground truth.


```python
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(row):
    """Compute cosine similarity between predicted and actual embeddings."""
    actual_vec = np.array(row["Actual_Embed"]).reshape(1, -1)
    predicted_vec = np.array(row["Predicted_Embed"]).reshape(1, -1)
    return cosine_similarity(actual_vec, predicted_vec)[0][0]

# Apply similarity computation
df_baseline_eval["Similarity"] = df_baseline_eval.apply(compute_similarity, axis=1)

```

### 4.4 Define Matching Based on Similarity Threshold
📌 What we're doing:
Instead of checking if the labels are exactly the same, we’ll say a prediction is “Correct” if similarity is above a threshold — say 0.8 (you can tune this later).


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Threshold for match ---
SIMILARITY_THRESHOLD = 0.8
df_baseline_eval["Match"] = df_baseline_eval["Similarity"] >= SIMILARITY_THRESHOLD

# --- Step 1: Semantic accuracy ---
semantic_accuracy = df_baseline_eval["Match"].mean()
print(f"🧠 Embedding-based Semantic Accuracy: {semantic_accuracy:.2%}")

# --- Step 2: Confusion matrix with similarity values ---
confusion_df = (
    df_baseline_eval.groupby(["output", "Prediction"])["Similarity"]
    .mean()
    .unstack(fill_value=0)
)

plt.figure(figsize=(12, 8))
sns.heatmap(confusion_df, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
plt.title("🔍 Confusion Matrix (Average Embedding Similarity)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Step 3: Precision / Recall / F1 based on semantic match ---
# Binary classification (Match = Correct)
precision = precision_score(df_baseline_eval["Match"], [True]*len(df_baseline_eval))
recall = recall_score(df_baseline_eval["Match"], [True]*len(df_baseline_eval))
f1 = f1_score(df_baseline_eval["Match"], [True]*len(df_baseline_eval))

print(f"📊 Precision: {precision:.2%}")
print(f"📊 Recall:    {recall:.2%}")
print(f"📊 F1 Score:  {f1:.2%}")

```

**Manual Evaluation for Individual Sample**


```python
sample_idx = 1

row = df_baseline_eval.iloc[sample_idx]
text = row["textInput"]
actual = row["output"]
predicted = row["Prediction"]
similarity = row["Similarity"]
match = row["Match"]

print(f"📝 Transcription:\n{text[:500]}...\n")  # Truncated for readability
print(f"✅ Actual Label:     {actual}")
print(f"🤖 Predicted Label:  {predicted}")
print(f"📐 Cosine Similarity: {similarity:.3f}")
print("🎯 Match:", "✅ Correct" if match else "❌ Incorrect")

```


```python

```

## 5. Fine-Tuning and Model Comparison



### 5.1 🔧 Fine-Tuning Gemini with Medical Transcription Data

In this step, we fine-tune the `gemini-1.5-flash-001` model using our medical transcription dataset.

We're using Google's **parameter-efficient fine-tuning (PEFT)** approach. This technique updates only a small number of model parameters (adapters), making training faster and more resource-efficient, while still enabling the model to adapt to our specific task — in this case, classifying medical transcriptions by specialty.

#### 📌 Key Parameters:
- **Base model**: `gemini-1.5-flash-001-tuning` — the fine-tunable version of Gemini Flash.
- **Training data**: A JSON-style dictionary containing a list of `{"textInput", "output"}` pairs.
- **Batch size**: 16 — number of samples processed together during each step.
- **Epochs**: 2 — each sample will be seen twice during training (for quick testing, can increase later).

Once the tuning job is submitted, we store the `model_id` so we can track and later evaluate or use the tuned model.

_Note: Tuning can take a few minutes to over an hour depending on load, so be patient or use a previously tuned model if available._



```python
# from google import genai
# from google.genai import types

tune_op = client.tunings.tune(
    base_model="models/gemini-1.5-flash-001-tuning",
    training_dataset=tune_data,
    config=types.CreateTuningJobConfig(
        tuned_model_display_name="medical-text-classifier",
        batch_size=16,
        epoch_count=2,  # start low for test run
    ),
)
model_id = tune_op.name
print("Tuning started:", model_id)

```

### Monitor Progress


```python
# model_id = "tunedModels/medicaltranscriptionclassifier-pnwuvgdln" ## data had so many labels
model_id = "tunedModels/medicaltextclassifier-matn7qfwnz0j" ## data had two labels 'urology' and 'nephrology'
model_status = client.tunings.get(name=model_id)
print(model_status.state)
# client.tunings.get(name=model_id).state
```

### 5.2 Use Your Tuned Model for Prediction
You just need to update your predict_label() function to call the tuned model.
And run Predictions on df_test


```python
# model_id = "tunedModels/medicaltranscriptionclassifier-pnwuvgdln" ## data had so many labels
model_id = "tunedModels/medicaltextclassifier-matn7qfwnz0j" ## data had two labels 'urology' and 'nephrology'
TUNED_MODEL_ID = model_id  # Replace with your actual ID

@retry.Retry(predicate=is_retriable)
def predict_label_tuned(text: str) -> str:
    response = client.models.generate_content(
        model=TUNED_MODEL_ID,
        contents=text
    )
    rc = response.candidates[0]
    return rc.content.parts[0].text.strip() if rc.finish_reason.name == "STOP" else "(error)"


df_tuned_eval = df_test.copy()
df_tuned_eval["Prediction"] = df_tuned_eval["textInput"].progress_apply(predict_label_tuned)


```

### 5.3 Embedding-Based Semantic Evaluation

We used vector embeddings to measure how closely the predicted labels align with the true labels. The workflow includes:

- Step 1: Embed Actual and Predicted Labels
- Step 2: Compute Cosine Similarity
- Step 3: Define a Match Threshold
- Step 4: Semantic Accuracy Score


```python

df_tuned_eval["Actual_Embed"] = df_tuned_eval["output"].progress_apply(embed_fn)
df_tuned_eval["Predicted_Embed"] = df_tuned_eval["Prediction"].progress_apply(embed_fn)


# Calculate Similarity Scores
df_tuned_eval["Similarity"] = df_tuned_eval.apply(compute_similarity, axis=1)


# Define Match by Similarity Threshold
SIMILARITY_THRESHOLD = 0.8
df_tuned_eval["Match"] = df_tuned_eval["Similarity"] >= SIMILARITY_THRESHOLD


# Semantic Accuracy
semantic_accuracy = df_tuned_eval["Match"].mean()
print(f"🧠 Tuned Model Semantic Accuracy: {semantic_accuracy:.2%}")

```

**📊 Confusion Matrix with Similarity Weights**


We also generate a confusion matrix that shows average cosine similarity between each actual vs. predicted label pair:


```python
# Confusion Matrix Heatmap (Similarity-Weighted)
import seaborn as sns
import matplotlib.pyplot as plt

confusion_df = (
    df_tuned_eval.groupby(["output", "Prediction"])["Similarity"]
    .mean()
    .unstack(fill_value=0)
)

plt.figure(figsize=(12, 8))
sns.heatmap(confusion_df, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
plt.title("🔍 Tuned Model Confusion Matrix (Embedding Similarity)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```


```python
# Precision, Recall, F1 (Semantic Match as Proxy for Correctness)
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(df_tuned_eval["Match"], [True]*len(df_tuned_eval))
recall = recall_score(df_tuned_eval["Match"], [True]*len(df_tuned_eval))
f1 = f1_score(df_tuned_eval["Match"], [True]*len(df_tuned_eval))

print(f"📊 Precision: {precision:.2%}")
print(f"📊 Recall:    {recall:.2%}")
print(f"📊 F1 Score:  {f1:.2%}")


```

## 6. Compare and validate the Tuned Model

Here are some challenging test examples — intentionally ambiguous between Nephrology (kidneys, renal function) and Urology (urinary tract, bladder, prostate). 

These are crafted to make it tricky even for humans:


```python
from google.api_core import retry
from google.genai import types
import pandas as pd

# Define model IDs
BASELINE_MODEL_ID = "models/gemini-1.5-flash-001"
TUNED_MODEL_ID = "tunedModels/medicaltextclassifier-matn7qfwnz0j"


# Define system instruction for the baseline model for classification
system_instruct = """
You are a classification service. You will be passed input that represents
a medical transcription,
and you must respond with the category it belongs to.
"""




# Retry handler
is_retriable = lambda e: isinstance(e, genai.errors.APIError) and e.code in {429, 503}

@retry.Retry(predicate=is_retriable)
def get_prediction(model_id, prompt_text):
    if model_id == BASELINE_MODEL_ID:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruct
            )
        )
    else:
        # For tuned models, no need for system prompt
        response = client.models.generate_content(
            model=model_id,
            contents=prompt_text
        )
    
    return response.text.strip()


# Define prompts and actual labels
test_cases = [
    {
        "Prompt": "The patient presents with recurrent flank pain and hematuria. Imaging showed a 7 mm calculus...",
        "FullPrompt": "The patient presents with recurrent flank pain and hematuria. Imaging showed a 7 mm calculus in the left ureter near the vesicoureteral junction. Creatinine levels are slightly elevated. History of hypertension and type 2 diabetes.",
        "Actual": "nephrology"
    },
    {
        "Prompt": "Patient reports difficulty initiating urination, dribbling, and mild lower back pain...",
        "FullPrompt": "Patient reports difficulty initiating urination, dribbling, and mild lower back pain. PSA levels normal. No signs of infection. Ultrasound indicates mild hydronephrosis and possible bladder outlet obstruction.",
        "Actual": "urology"
    },
    {
        "Prompt": "A 58-year-old male with a history of chronic kidney disease stage 3, presents with urgency...",
        "FullPrompt": "A 58-year-old male with a history of chronic kidney disease stage 3, presents with urgency and frequency. Urinalysis shows microalbuminuria and trace blood. Renal ultrasound normal. Referred for urologic evaluation.",
        "Actual": "nephrology"
    },
]

# Collect results
results = []

for case in test_cases:
    baseline_pred = get_prediction(BASELINE_MODEL_ID, case["FullPrompt"])
    tuned_pred = get_prediction(TUNED_MODEL_ID, case["FullPrompt"])
    
    results.append({
        "Prompt (abbreviated)": case["Prompt"],
        "Actual": case["Actual"],
        "Baseline Prediction": baseline_pred,
        "Baseline Match": "✅" if baseline_pred.lower() == case["Actual"].lower() else "❌",
        "Tuned Prediction": tuned_pred,
        "Tuned Match": "✅" if tuned_pred.lower() == case["Actual"].lower() else "❌",
    })

# Convert to DataFrame and show as table
df_results = pd.DataFrame(results)
import IPython.display as display
display.display(df_results)


```


```python

```

---

# 🤖 Phase 2: MedBot – LangGraph Chatbot for Medical Transcription Classification

This phase focuses on building an interactive medical assistant chatbot named **MedBot** using [LangGraph](https://www.langchain.com/langgraph). MedBot is designed to assist users with **nephrology** and **urology** cases by classifying user-provided clinical summaries.

---

### 🧠 What MedBot Can Do (Current Capabilities)

- **Conversational Input Handling**: MedBot chats naturally with users to collect patient symptoms, diagnoses, and clinical details.
- **Medical Classification**: Once enough information is gathered, MedBot uses a **Gemini 1.5 model** (prompt-engineered or fine-tuned) to classify the case into one of the following:
  - **Nephrology**
  - **Urology**
  - **Other**
- **Scoped Dialogue Management**: If the case is outside its scope (e.g., classified as "Other"), it politely redirects the user and suggests consulting a provider.

---

### 🛠️ GenAI Techniques Used

- **Prompt Engineering**: Clear system instructions guide MedBot’s tone, behavior, and tool invocation.
- **LangGraph Modular Design**:
  - **State management** for conversation history and tool results.
  - **Conditional routing** between nodes like the chatbot, tool classifier, and human.
- **LLM Tool Invocation**: Tools (like `classify_transcription`) are triggered based on user input patterns.

---

### ✅ Summary

This modular chatbot architecture serves as a foundation for a more intelligent medical assistant system. It demonstrates structured LLM interaction with LangGraph, real-time tool invocation, and safe domain-specific communication—all key requirements for a modern GenAI assistant in healthcare support.

---


We need to restart the kernell and start running from here again.

At this point we have everything that we have:
- embedded transcriptions of all samples with 'Urology' or 'Nephrology' Category.
- fine-tuned model that classify any given text.
- original transcription data.

## 1. Load and Configure Environment

This step loads:
- The tuned Gemini model ID
- The dataframe with **20 samples per class**
- Precomputed Gemini embeddings for retrieval



```python
import os
from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```


```python
# Remove conflicting packages from the Kaggle base environment.
!pip uninstall -qqy kfp jupyterlab libpysal thinc spacy fastai ydata-profiling google-cloud-bigquery google-generativeai
# Install langgraph and the packages used in this lab.
!pip install -qU 'langgraph==0.3.21' 'langchain-google-genai==2.1.2' 'langgraph-prebuilt==0.1.7'
```


```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google as genai
from google.api_core import retry
from google.genai import types
from google.api_core import exceptions
from sklearn.metrics.pairwise import cosine_similarity
from google.api_core import retry


# Set model IDs
TUNED_MODEL_ID = "tunedModels/medicaltextclassifier-matn7qfwnz0j"
EMBEDDING_MODEL = "models/text-embedding-004"


## embeddings of all transcriptions from Urology or Nephrology class
df = pd.read_csv('/kaggle/input/dembeddings-all-uro-nephro-logy-transcriptions/df_with_embeddings_all_uro_nephro_logy_transcriptions.csv', index_col=0)
```

## 2. Define MedBot State and Welcome Message

This state definition uses `TypedDict` and LangGraph's `add_messages` annotation to preserve message history across conversation turns. We also define the initial welcome message the bot will display when it starts.

This step sets up:
- `MedState`: conversation state
- `WELCOME_MSG`: opening line from MedBot



```python
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages.ai import AIMessage


# ✅ MedBot state definition (inspired by BaristaBot)
class MedState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    predicted_category: str
    retrieved_case: dict
    ground_response: str
    finished: bool

# ✅ System instruction for MedBot
MEDBOT_SYSINT = (
    "system",
    "You are MedBot, a specialized assistant for nephrology and urology cases.\n\n"
    "When the user provides a case description, ALWAYS call the `classify_transcription` tool "
    "with their message to determine if the issue is related to nephrology, urology, or something else.\n\n"
    "Only after calling the tool and receiving the classification, respond to the user appropriately:\n"
    "- If the issue is classified as urology or nephrology, say that it seems related to that field, and offer to retrieve similar cases or accept more details.\n"
    "- If the issue is classified as 'Other', say that it's outside your scope and offer to help find a provider nearby.\n\n"
    "Do not guess the category yourself — always invoke the `classify_transcription` tool first.\n\n"
    "You must call the tool every time the user describes a patient case."
)



# ✅ Updated welcome message
WELCOME_MSG = "🩺 MedBot is ready. Describe a patient case related to nephrology or urology (or type 'exit' to quit)."

```

## 3. Define the MedBot Chat Node and Build the Initial Graph

In this step, we implement the core chatbot logic for a single conversational turn. The LangGraph graph will start with this chatbot node and terminate afterward (we’ll add human input and looping in the next step).

- `medbot()` uses the Gemini model to respond to messages in `state["messages"]`
- Messages are prepended with the `MEDBOT_SYSINT` system instruction
- The graph starts from `START`, runs the chatbot node, and ends at `END`



```python
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini LLM (same as BaristaBot)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def medbot(state: MedState) -> MedState:
    """MedBot logic for now is just to respond using the Gemini model."""
    history = [MEDBOT_SYSINT] + state["messages"]
    return {"messages": [llm.invoke(history)]}

# Set up initial graph
graph_builder = StateGraph(MedState)
graph_builder.add_node("medbot", medbot)
graph_builder.add_edge(START, "medbot")
med_graph = graph_builder.compile()

```


```python
from IPython.display import Image, display

Image(med_graph.get_graph().draw_mermaid_png())
```

✅ You can now test a single-turn interaction with:


```python
state = med_graph.invoke({"messages": ['Hello']})
for msg in state["messages"]:
    print(f"{type(msg).__name__}: {msg.content}")

```

## 4. Add Human Node and Interaction Loop

We add a `human` node that:

- Displays the assistant's last message.
- Collects user input from the command line.
- Terminates if the user types an exit command like `exit` or `q`.

We also update the MedBot node to send a welcome message if it's the start of the conversation.



```python
# Human node: responds to MedBot and allows input
def human_node(state: MedState) -> MedState:
    last_msg = state["messages"][-1]
    print("🤖 MedBot:", last_msg.content)
    
    user_input = input("👤 You: ")

    if user_input.lower().strip() in {"exit", "quit", "q"}:
        state["finished"] = True

    return state | {"messages": [("user", user_input)]}

# Enhanced MedBot: sends welcome message if no prior messages
def medbot_with_welcome(state: MedState) -> MedState:
    if state["messages"]:
        output = llm.invoke([MEDBOT_SYSINT] + state["messages"])
    else:
        output = {"content": "🩺 MedBot is ready. Describe a patient case related to nephrology or urology (or type 'exit' to quit)."}
    return state | {"messages": [output]}

```

## 5. Define Routing Logic and Loop the Conversation

We define a conditional edge:
- If the user says "exit", go to END.
- Otherwise, loop back to MedBot for more interaction.



```python
from typing import Literal

# Control flow logic: loop or exit
def maybe_exit(state: MedState) -> Literal["medbot", "__end__"]:
    return END if state.get("finished") else "medbot"

# Build the full graph with loop
graph_builder = StateGraph(MedState)
graph_builder.add_node("medbot", medbot_with_welcome)
graph_builder.add_node("human", human_node)

graph_builder.add_edge(START, "medbot")
graph_builder.add_edge("medbot", "human")
graph_builder.add_conditional_edges("human", maybe_exit)

medbot_loop_graph = graph_builder.compile()

```


```python
from IPython.display import Image, display

Image(medbot_loop_graph.get_graph().draw_mermaid_png())
```

**Run MedBot**
Now we can run MedBot in a loop. Type a description of a case, and MedBot will respond.

Type `exit` to end the session.



```python
# Uncomment to run
state = medbot_loop_graph.invoke({"messages": ['Hi']},
                                 config={"recursion_limit": 50})

```


```python

```

## 6. Add a Classifier Tool Node

We define a `classify_transcription` tool.

This tool will:
- Use your fine-tuned Gemini model to classify a patient transcription.
- Predict whether it relates to **Nephrology** or **Urology**.

We'll wrap this tool using LangGraph's `ToolNode` so it can be invoked automatically when MedBot detects the need.


Define the Tool


```python
MEDBOT_SYSINT = (
    "system",
    "You are MedBot, an AI assistant that supports patients and providers with nephrology and urology cases only.\n\n"

    "Here is how you should interact:\n\n"

    "🩺 PHASE 1 — INTRODUCTION:\n"
    "- Greet the user and invite them to describe the patient case.\n"
    "- Examples: symptoms, diagnoses, labs, imaging findings, etc.\n\n"

    "📝 PHASE 2 — COLLECT HISTORY:\n"
    "- Keep a running summary of all relevant medical information as `patient_summary`.\n"
    "- After each message, if the user provides partial information, ask clarifying questions.\n"
    "- Do NOT call any tools until the user confirms they have shared everything (e.g., says 'that's all', 'no more info', etc).\n\n"

    "🧠 PHASE 3 — CLASSIFY:\n"
    "- Once the patient summary is complete, call the `classify_transcription` tool using the full `patient_summary`.\n"
    "- Use the exact tool call format below:\n"
    "```tool_code\n"
    "classify_transcription(transcription=\"...patient_summary...\")\n"
    "```\n"
    "- Wait for the classification result before continuing.\n\n"

    "🧭 PHASE 4 — BRANCH BY CATEGORY:\n"
    "- If classification is `Urology` or `Nephrology`:\n"
    "  • Say: 'Based on your summary, this seems related to [CATEGORY].'\n"
    "  • Ask: 'Would you like me to retrieve a similar case from my documentation?'\n"
    "  • If user says yes, call the future retrieval tool using `patient_summary`.\n"
    "  • After retrieving a case, offer to help them find a local [CATEGORY] specialist.\n\n"
    "- If classification is `Other`:\n"
    "  • Say: 'This doesn’t seem to fall within nephrology or urology.'\n"
    "  • Offer to help them find a general doctor or hospital in their area.\n\n"

    "🌐 PHASE 5 — LOCATION HELP:\n"
    "- If user wants to find help, ask for their ZIP code or city.\n"
    "- Then call the ground search tool with the location and specialty.\n"
    "- Present the results politely.\n\n"

    "🚫 SAFETY AND SCOPE:\n"
    "- Never provide diagnoses, treatment recommendations, or interpret lab/imaging.\n"
    "- Always remind the user to consult a real physician for decisions.\n"
    "- Stay within the nephrology/urology domain at all times.\n\n"

    "💬 TONE:\n"
    "- Be polite, helpful, concise, and professional.\n"
    "- Always thank the user for information and invite clarification.\n"
    "- Avoid hallucinating or guessing — stick to instructions and tools only.\n"
)

```


```python
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

class MedState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    predicted_category: str
    patient_summary: str
    retrieved_case: dict
    ground_response: str
    finished: bool

```


```python
# Your fine-tuned model
TUNED_MODEL_ID = "tunedModels/medicaltextclassifier-matn7qfwnz0j"
BASELINE_MODEL_ID = "models/gemini-2.0-flash"

from google.api_core import retry
from google.genai import types

# Classification instruction prompt
classification_prompt = """
You are a classification service. You will be passed input that represents a medical transcription, and you must respond with the category it belongs to.
Valid categories are: Urology, Nephrology, or Other.
Only respond with the category name. Do not include explanations.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_core.tools import tool

class ClassificationInput(BaseModel):
    transcription: str

@tool(args_schema=ClassificationInput)
def classify_transcription(transcription: str) -> str:
    """Classify transcription as Urology, Nephrology, or Other."""
    return "placeholder"




classifier_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# ✅ Bind your tool so the LLM can return tool_calls
llm_with_tools = classifier_llm.bind_tools([classify_transcription])




```

## 7. Register the Classifier Tool with LangGraph
What we're doing:

Wrap the classify_transcription function into a ToolNode, just like get_menu in BaristaBot.

Bind the tool to the Gemini model (llm_with_tools) so it knows the tool exists and can call it.

Update your chatbot to use this new llm_with_tools.

 Add Tool Node


```python
def classifier_node(state: MedState) -> MedState:
    print("🛠️ Classifier node triggered.")
    
    tool_call = state["messages"][-1].tool_calls[0]
    input_text = tool_call["args"]["transcription"]

    # 🧠 Use the fine-tuned model directly
    prompt = """You are a classification service. 
    You will be passed input that represents a medical transcription, and you must respond with the category it belongs to.
    Valid categories: Nephrology, Urology, or Other.
    Only respond with the category name. No explanations."""
    response = llm_with_tools.invoke([("system", prompt), ("user", input_text)])
    label = response.content.strip()

    state["predicted_category"] = label

    if label.lower() in {"urology", "nephrology"}:
        reply = (
            f"🧠 Based on your input, this seems related to **{label}**.\n"
            "Would you like me to pull up a similar patient transcription, or do you have more clinical details to add?"
        )
    else:
        reply = (
            "🧠 This issue doesn’t seem related to nephrology or urology.\n"
            "I'm only trained to support those specialties.\n"
            "Would you like me to help you find a doctor or clinic in your area?"
        )

    return state | {
        "messages": [
            ToolMessage(
                content=label,
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            ),
            AIMessage(content=reply)
        ]
    }

```

## 8. Bind Tools to MedBot, Add Classifier Tool Node to Graph
We'll update the graph to include the tool node and routing logic.

To allow the LLM to call tools like `classify_transcription`, we bind it to the model.



```python
from langchain_core.messages import AIMessage, ToolMessage

def medbot_with_tools(state: MedState) -> MedState:
    defaults = {
        "predicted_category": None,
        "retrieved_case": None,
        "ground_response": None,
        "finished": False
    }
    
    history = state.get("messages", [])
    
    # 🩺 First message? Show welcome
    if not history:
        return defaults | state | {"messages": [AIMessage(content=WELCOME_MSG)]}
    
    # 🛠 If previous message was a tool result, interpret it
    last_msg = history[-1]
    if isinstance(last_msg, ToolMessage) and last_msg.name == "classify_transcription":
        classification = last_msg.content.strip()
        state["predicted_category"] = classification

        if classification.lower() in {"urology", "nephrology"}:
            reply = (
                f"🧠 Based on the information, this case seems related to **{classification}**.\n"
                "Would you like me to retrieve a similar patient transcription from my documentation?"
            )
        else:
            reply = (
                "🧠 This doesn’t seem related to nephrology or urology.\n"
                "I'm only trained to support those specialties.\n"
                "Would you like me to help you find a doctor or clinic in your area?"
            )
        return state | {"messages": [AIMessage(content=reply)]}

    # 👂 Otherwise continue conversation and tool use with tool-bound LLM
    output = llm_with_tools.invoke([MEDBOT_SYSINT] + history)
    return state | {"messages": [output]}

```


```python
from typing import Literal

import re


def maybe_route_to_classifier(state: MedState) -> Literal["classifier", "human"]:
    last_msg = state["messages"][-1]
    return "classifier" if hasattr(last_msg, "tool_calls") and last_msg.tool_calls else "human"





graph_builder = StateGraph(MedState)

graph_builder.add_node("medbot", medbot_with_welcome)
graph_builder.add_node("human", human_node)
graph_builder.add_node("classifier", classifier_node)

graph_builder.add_edge(START, "medbot")

# 🧠 Route to tool or user based on tool_calls
graph_builder.add_conditional_edges("medbot", maybe_route_to_classifier)

# 🛠 Tool always routes back to bot
graph_builder.add_edge("classifier", "medbot")

# 👤 Human either exits or loops back
graph_builder.add_conditional_edges("human", maybe_exit)

medbot_graph_manual_classifier = graph_builder.compile()

```


```python
from IPython.display import Image, display

Image(medbot_graph_manual_classifier.get_graph().draw_mermaid_png())
```

### 🧪 Sample Prompts for Testing the Chatbot
**🟢 Nephrology-related:**
- "The patient has elevated creatinine levels and persistent proteinuria over the last two months."

- "Patient presents with chronic kidney disease stage 3 and reports fatigue and edema."

- "Blood work reveals abnormal GFR and microalbuminuria; patient has a history of diabetes and hypertension."

**🔵 Urology-related:**
- "The patient reports difficulty urinating, lower abdominal pressure, and increased frequency at night."

- "Ultrasound shows an enlarged prostate with post-void residual urine volume of 150 mL."

- "Reports burning sensation during urination, urgency, and a recent UTI treated with antibiotics."

**⚪ Mixed or borderline case:**
- "58-year-old male with hypertension presents with hematuria and mild flank pain; imaging shows mild hydronephrosis."

**🔴 Unrelated (non-nephrology/urology):**
- "The patient is experiencing chest tightness and pain radiating to the left arm, especially during exertion."
(Expected: The model might still try to assign one of the two known categories unless specifically trained to say “neither.”)


```python
# Uncomment to run
state = medbot_graph_manual_classifier.invoke({"messages": ['Hi']},
                                 config={"recursion_limit": 50})

```


```python
for msg in state["messages"]:
    print(type(msg), getattr(msg, "name", ""), msg.content)

```
