from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import recursive text splitter

text = """"  # Input text to split
An MCP (Model Context Protocol) Server is a backend service that follows the MCP standard to let AI models (like LLMs) securely interact with external tools, data, and systems. Instead of granting unrestricted access to the internet, databases, or APIs, the MCP server acts as a controlled bridge, exposing only specific resources, actions, or data sources in a standardized way. This ensures that the model can only access what is explicitly permitted, reducing risks of misuse or overreach.

In practice, an MCP server enforces security, consistency, and interoperability by setting strict boundaries around what data and operations are available. This makes it a safe connector that enables AI assistants to work with external systems—such as databases, APIs, or apps—without compromising control. In short, it provides structured, permissioned access that empowers LLMs to be more useful and reliable while protecting sensitive resources.
"""

splitter = RecursiveCharacterTextSplitter(  # Create recursive text splitter
    chunk_size = 100,  # Max size of each chunk
    chunk_overlap = 15  # Overlap between chunks
)

chunks = splitter.split_text(text)  # Split text into chunks

print(len(chunks))  # Print number of chunks
print(f"Total chunks: {len(chunks)}\n")  # Print total chunks with message
for i, chunk in enumerate(chunks, 1):  # Loop through chunks
    print(f"--- Chunk {i} ---")  # Print chunk number
    print(chunk)  # Print chunk content
