import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def chunk_text(text, max_length=1000):
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks

def summarize_text_mistral(text, video_info, model, tokenizer):
    prompt = (
        f"Summarize the following video transcript chunk in a coherent and detailed manner. "
        f"Highlight key points and maintain the flow of the narrative. Include information about the video titled '{video_info['title']}' by '{video_info['author']}' with the following description: '{video_info['description']}':\n\n"
        f"{text}\n\nSummary:"
    )
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    summary_ids = model.generate(inputs, max_new_tokens=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = summary.split("Summary:")[-1].strip()
    return summary

def query_model(question, context, model, tokenizer):
    prompt = f"{question}\n\nContext:\n{context}\n\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    answer_ids = model.generate(inputs, max_new_tokens=200, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()
    return answer
