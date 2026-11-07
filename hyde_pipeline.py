from haystack import Pipeline

from hyde_components import (
    prompt_builder,
    generator,
    RepeatedGenerator,
    adapter,
    embedder,
    HypotheticalDocumentEmbedder,
)

pipeline = Pipeline()
pipeline.add_component(name="prompt_builder", instance=prompt_builder)
pipeline.add_component(name="generator", instance=RepeatedGenerator(generator, n=5))
pipeline.add_component(name="adapter", instance=adapter)
pipeline.add_component(name="embedder", instance=embedder)
pipeline.add_component(name="hyde", instance=HypotheticalDocumentEmbedder())

# question → 5 hypothetical answers → Documents → embeddings → one average vector
pipeline.connect("prompt_builder.prompt", "generator.messages")
pipeline.connect("generator.replies", "adapter.answers")
pipeline.connect("adapter.output", "embedder.documents")
pipeline.connect("embedder.documents", "hyde.documents")

if __name__ == "__main__":
    query = "What should I do if I have a fever?"
    result = pipeline.run(data={"prompt_builder": {"question": query}}, include_outputs_from={'embedder'})

    hyde_vector = result["hyde"]["hypothetical_embedding"]
    print(f"Query: {query}")
    print(f"HyDE embedding length: {len(hyde_vector)}")
    print(f"First 5 values: {hyde_vector[:5]}")
    
    documents = result["embedder"]["documents"]
    print(f"\nNumber of hypothetical answers: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        print(f"\nAnswer {i}: {doc.content[:200]}...")