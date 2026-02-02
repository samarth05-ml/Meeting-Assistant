import torch
import os
import gradio as gr
from transformers import pipeline

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


# -------- IBM watsonx Configuration --------
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

params = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}

LLAMA_model = Model(
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    credentials=my_credentials,
    params=params,
    project_id="skills-network"
)

llm = WatsonxLLM(LLAMA_model)


# -------- Prompt Template --------
template = """
<s><<SYS>>
List the key points with details from the context:
[INST] The context : {context} [/INST]
<</SYS>>
"""

prompt_template = PromptTemplate(
    input_variables=["context"],
    template=template
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)


# -------- Speech to Text + Analysis --------
def transcript_audio(audio_file):
    # Speech-to-text using Whisper
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30
    )

    transcript_text = pipe(audio_file, batch_size=8)["text"]

    # LLM analysis
    analyzed_output = llm_chain.run(transcript_text)

    return analyzed_output


# -------- Gradio Interface --------
audio_input = gr.Audio(
    sources="upload",
    type="filepath",
    label="Upload Meeting Audio"
)

output_text = gr.Textbox(
    label="Meeting Key Points"
)

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="AI-Powered Meeting Assistant",
    description="Upload a meeting audio file to extract key discussion points"
)

# -------- Launch App --------
iface.launch(
    server_name="0.0.0.0",
    server_port=7860
)
