# 🏥 MedBot: A GenAI-Powered Assistant for Classifying and Responding to Medical Transcriptions

## 📘 Project Overview

This project demonstrates how **Generative AI techniques** can be combined to build a real-world assistant for healthcare support. The assistant, called **MedBot**, is designed to understand medical case descriptions, classify them into medical specialties, and offer relevant next steps—such as retrieving similar records or suggesting nearby providers.

The solution showcases both **model fine-tuning** and **tool-augmented conversation design** using **LangGraph** and **Gemini API**.

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

