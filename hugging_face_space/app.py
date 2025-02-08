import gradio as gr
from huggingface_hub import InferenceClient
from transformers import pipeline
import random
import time

# """
# For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
# """

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
def respond(
    message,
    history: list[tuple[str, str]],
    system_message="Please do not speak in bullet point format, please only speak in conversation form.",
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
):
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

on_load = """
async()=>{
    console.log("HELLO WORLD");
}
"""

demo = gr.ChatInterface(
    respond,
    js = on_load,
    # additional_inputs=[
    #     gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
    #     gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
    #     gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
    #     gr.Slider(
    #         minimum=0.1,
    #         maximum=1.0,
    #         value=0.95,
    #         step=0.05,
    #         label="Top-p (nucleus sampling)",
    #     ),
    # ],
)

if __name__ == "__main__":
    demo.launch()
