import gradio as gr

# Simple greeting function
def greet(name):
    return "Hello " + name + "!"

# Gradio interface for testing
demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
)

# Launch the app on all network interfaces at port 7860
demo.launch(server_name="0.0.0.0", server_port=7860)
