from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
    Please create a summary of the research topic titled "{paper_input}".

    Write the explanation in the following way:
    - Style: {style_input}
    - Length: {length_input}

    1. Provide key concepts, potential applications, and current trends in this research area.
    2. Include mathematical details or equations only if they are commonly associated with this field.
    3. Use analogies to make complex ideas more relatable.

    If information is highly speculative, make reasonable assumptions instead of saying 'insufficient information'.
    Ensure the summary is clear, accurate, and aligned with the provided style and length.
    """,
    input_variables=["length_input", "paper_input", "style_input"],
    validate_template=True,
)

template.save("./template.json")
