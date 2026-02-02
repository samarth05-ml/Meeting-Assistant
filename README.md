# Meeting-Assistant
In this project , I'm using OpenAI whisper and LlaMA 2 LLM
Section	Content
Project Name	AI-Powered Meeting Assistant
Description	An AI application that converts meeting audio into text and extracts meaningful insights using Speech-to-Text and Large Language Models.
Main Features	• Speech-to-Text using Whisper
• Key point extraction using LLM (LLaMA)
• Simple web UI with Gradio
Technologies Used	Python, Hugging Face Transformers, OpenAI Whisper, IBM watsonx (LLaMA), LangChain, Gradio, PyTorch
Project Structure	speech_analyzer.py – Main pipeline
speech2text_app.py – Speech-to-Text module
simple_llm.py – LLM test
simple_speech2text.py – Whisper test
How It Works	1. Upload meeting audio
2. Convert speech to text
3. Analyze text using LLM
4. Extract key points
5. Display results
Use Cases	Meeting summarization, Lecture notes, Interview analysis, Productivity tools

⚠️ NOTE
This project was developed and tested in an IBM Skills Network / lab environment. Certain values such as project_id="skills-network" and authentication setup are environment-specific and may need modification when running locally or deploying elsewhere.
