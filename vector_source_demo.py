from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import gradio as gr



reader = SimpleDirectoryReader( 
    input_files=["~/Documents/slm_flow/data/marketing_and_sales_IX.pdf"]
)
documents = reader.load_data()



system_prompt = "You are a senior marketing executive. Your goal is to answer questions as accurately as possible based on the instructions and context provided."

query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")



model_path = "/home/andreas/Documents/tiny_titan"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name='/home/andreas/Documents/tiny_titan',
    model_name='/home/andreas/Documents/tiny_titan',
    #device_map="cuda",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.bfloat16}
)
# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()

def predict(input, history):
  response = query_engine.query(input)
  return str(response)

gr.ChatInterface(predict).launch(share=True)

