from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# Credentials for IBM watsonx (using IBM lab environment)
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

# Parameters for the LLM model
params = {
    GenParams.MAX_NEW_TOKENS: 700,  # Maximum tokens to generate
    GenParams.TEMPERATURE: 0.1,     # Controls randomness of output
}

# Initialize the LLaMA model hosted on IBM watsonx
LLAMA2_model = Model(
    model_id='meta-llama/llama-3-2-11b-vision-instruct',
    credentials=my_credentials,
    params=params,
    project_id="skills-network",
)

# Wrap the model in WatsonxLLM for LangChain compatibility
llm = WatsonxLLM(LLAMA2_model)

# Test the LLM with a sample prompt
print(llm("How to read a book effectively?"))
