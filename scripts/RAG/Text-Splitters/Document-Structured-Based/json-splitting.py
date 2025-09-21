import json  # Import JSON module for handling JSON data
from langchain_text_splitters import RecursiveJsonSplitter  # type: ignore # Import JSON splitter from LangChain

# JSON text containing personal information
json_text = """
{
  "name": "Siddhartha Shakya",
  "role": "AI & Data Enthusiast",
  "location": "Nepal",
  "skills": {
    "AI & ML": ["Scikit-Learn", "Pandas", "NumPy"],
    "Programming": ["Python", "C"],
    "Databases": ["MySQL", "MongoDB"],
    "Web & APIs": ["FastAPI", "LangChain"],
    "AI Integration": ["MCP (Model Context Protocol)"],
    "Tools": ["Git", "Linux", "Ubuntu", "Docker", "VS Code"]
  },
  "interests": [
    "Sentiment analysis & NLP",
    "Data pipelines",
    "LangChain + RAG pipelines",
    "AI integrations with MCP",
    "Open-source projects"
  ],
  "contacts": {
    "email": "siddhuushakyaa@gmail.com",
    "github": "https://github.com/SiddhuShkya",
    "linkedin": "https://www.linkedin.com/in/siddhartha-shakya-5665a0236/",
    "leetcode": "https://leetcode.com/SiddhuShkya"
  }
}
"""

json_data = json.loads(json_text)  # Convert JSON string to Python dictionary

splitter = RecursiveJsonSplitter(  # Create a JSON splitter with chunk size limit
    max_chunk_size=200,
)

chunks = splitter.split_json(json_data)  # Split JSON data into smaller chunks

print(f"Total chunks: {len(chunks)}\n")  # Print total number of chunks
for i, chunk in enumerate(chunks, 1):  # Loop through each chunk with index
    print(f"--- Chunk {i} ---")  # Print chunk header
    print(chunk)  # Print chunk content
