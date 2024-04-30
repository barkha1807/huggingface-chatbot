from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []


def chatbot(message, history):
    history_string = "\n".join(conversation_history)

    inputs = tokenizer.encode_plus(history_string, message, return_tensors="pt")
    outputs = model.generate(**inputs)
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    conversation_history.append(message)
    conversation_history.append(response)

    return response

if __name__ == "__main__":
    # Create a Gradio chat interface
    demo_chatbot = gr.ChatInterface(chatbot, title="AI Chatbot", description="Enter your message here")
    demo_chatbot.launch()