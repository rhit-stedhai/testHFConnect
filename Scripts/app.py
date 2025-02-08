import gradio as gr
from huggingface_hub import InferenceClient
from transformers import pipeline
import random
import time

# """
# For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
# """
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# def respond(
#     message,
#     history: list[tuple[str, str]],
#     system_message="You are a friendly Chatbot who is taking on the role of a mental health counselor.",
#     max_tokens=512,
#     temperature=0.7,
#     top_p=0.95,
# ):
#     messages = [{"role": "system", "content": system_message}]
#     for val in history:
#         if val[0]:
#             messages.append({"role": "user", "content": val[0]})
#         if val[1]:
#             messages.append({"role": "assistant", "content": val[1]})
#     messages.append({"role": "user", "content": message})

#     response = ""
#     for message in client.chat_completion(
#         messages,
#         max_tokens=max_tokens,
#         stream=True,
#         temperature=temperature,
#         top_p=top_p,
#     ):
#         token = message.choices[0].delta.content
#         response += token
#         yield response

# on_load = """
# async()=>{
#     console.log("HELLO WORLD");
# }
# """

# demo = gr.ChatInterface(
#     respond,
#     js = on_load,
#     # additional_inputs=[
#     #     gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
#     #     gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
#     #     gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
#     #     gr.Slider(
#     #         minimum=0.1,
#     #         maximum=1.0,
#     #         value=0.95,
#     #         step=0.05,
#     #         label="Top-p (nucleus sampling)",
#     #     ),
#     # ],
# )

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history: list):
    response = "**That's cool!**"
    history.append({"role": "assistant", "content": ""})
    for character in response:
        history[-1]["content"] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
        sources=["microphone", "upload"],
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None, like_user_message=True)

if __name__ == "__main__":
    demo.launch()
