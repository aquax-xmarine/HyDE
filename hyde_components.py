from dotenv import load_dotenv
from numpy import array, mean

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack import component, Document
from haystack.components.converters import OutputAdapter
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
)
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

load_dotenv()

# Initializing standard Haystack components
generator = OpenAIChatGenerator(
    api_key=Secret.from_env_var("DEEPSEEK_API_KEY"),
    api_base_url="https://api.deepseek.com",
    model="deepseek-chat",
    generation_kwargs={"temperature": 0.75, "max_tokens": 400},
)

# DeepSeek only supports n=1 (one reply per call), so this component
# calls the generator n times and collects all the replies
@component
class RepeatedGenerator:
    def __init__(self, generator, n: int = 5):
        self.generator = generator
        self.n = n
 
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]):
        replies = []
        for _ in range(self.n):
            replies.extend(self.generator.run(messages=messages)["replies"])
        return {"replies": replies}

        
prompt_builder = ChatPromptBuilder(
    template=[
        ChatMessage.from_user(
            """Given a question, generate a paragraph of text that answers the question.    Question: {{question}}    Paragraph:""",
        ),
    ],
    required_variables="*",
)

# The ChatGenerator returns ChatMessage replies, so we read each reply's text.
# unsafe=True lets the adapter return actual Document objects instead of a string.
adapter = OutputAdapter(
    template="{{answers | build_doc}}",
    output_type=list[Document],
    custom_filters={"build_doc": lambda data: [Document(content=d.text) for d in data]},
    unsafe=True,
)

embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2",
)


# Adding one custom component that returns one, "average" embedding from multiple (hypothetical) document embeddings
@component
class HypotheticalDocumentEmbedder:
    @component.output_types(hypothetical_embedding=list[float])
    def run(self, documents: list[Document]):
        stacked_embeddings = array([doc.embedding for doc in documents])
        avg_embeddings = mean(stacked_embeddings, axis=0)
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist()}