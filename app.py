import os
import gradio as gr

from transformerEnFa.pipeline.stage_07_model_inference import ModelInferencePipeline

def inference_model(text):
    try:
        translator = ModelInferencePipeline()
        transaltion = translator.main(text)
        return transaltion
    except Exception as e:
        return f"Error Occurred! {e}"

def train_model():
    os.system("python main.py")
    return "Training successful !!"


iface_predict = gr.Interface(
    fn=inference_model,
    inputs=gr.Textbox(lines=10, placeholder="Enter text here..."),
    outputs=gr.Textbox(),
    title="English to Farsi Translator",
    description="This tool translates English text into Farsi. Enter your text and press submit to see the translation.",
    examples=["Hello, how are you?", "What is your name?", "This is a test sentence for translation."]

)

iface_train = gr.Interface(
    fn=train_model,
    inputs=[],
    outputs=gr.Textbox(),
    title="Train Model",
    description="Trigger model training."
)

iface = gr.TabbedInterface([iface_predict, iface_train], ["Inference", "Train"])

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8080)
