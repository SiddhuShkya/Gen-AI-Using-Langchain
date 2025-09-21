from langchain.text_splitter import RecursiveCharacterTextSplitter, Language  # type: ignore # Import text splitter for structured text

# Markdown text (profile-style content)
md = """
# [![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&width=800&color=FFFFFF&lines=👋+Hi%2C+I'm+Siddhartha+Shakya)](https://git.io/typing-svg)

## 🚀 About Me  
I'm an **AI & Data Enthusiast from Nepal 🇳🇵**, passionate about exploring the intersection of **artificial intelligence and data science**. I love **solving problems through coding** and enjoy working on projects that combine **mathematical thinking, creativity, and practical implementation**.  

## 🌱 Currently Learning  
- **LangChain & RAG** (building LLM-powered applications)  
- **FastAPI** (creating efficient and scalable APIs)  
- **Docker** (containerization & deployment)  
- **MCP (Model Context Protocol)** (next-gen AI application integration)  
- **Machine Learning & Deep Learning** (building smarter models)  

## 💡 Interests & Projects  
- Building ML-powered tools for **sentiment analysis & NLP**  
- Exploring **efficient data pipelines** and databases (MySQL, MongoDB)  
- Experimenting with **LangChain + RAG pipelines** for intelligent applications  
- Exploring **MCP** for integrating AI into real-world workflows  
- Passionate about **open-source** and **real-world applications of AI**  

## 🛠 Tech Stack  
- **AI & ML:** Scikit-Learn, Pandas, NumPy  
- **Programming:** Python, C  
- **Databases:** MySQL, MongoDB  
- **Web & APIs:** FastAPI, LangChain  
- **AI Integration:** MCP (Model Context Protocol)  
- **Tools & Platforms:** Git, Linux, Ubuntu, Docker, VS Code  

---

## 📊 Stats  

<div align="center">

<table>
  <tr>
    <td align="center">
      <!-- LeetCode Stats -->
      <img src="https://leetcard.jacoblin.cool/S_Shakya?theme=dark&font=Baloo%20Chettan%202&ext=contest" alt="LeetCode Stats" width="400" />
      <br/><br/>
      <a href="https://leetcode.com/S_Shakya">
        <img src="https://img.shields.io/badge/LeetCode-S_Shakya-orange?logo=leetcode&style=for-the-badge" alt="LeetCode Profile" />
      </a>
    </td>
    <td align="center">
      <!-- GitHub Stats -->
      <img src="https://github-readme-stats.vercel.app/api?username=SiddhuShkya&show_icons=true&include_all_commits=true&count_private=true&theme=github_dark&hide_border=true" height="150" alt="GitHub Stats" />
      <br/>
      <img src="https://github-readme-stats.vercel.app/api/top-langs?username=SiddhuShkya&layout=compact&langs_count=6&theme=github_dark&hide_border=true" height="150" alt="Top Languages" />
    </td>
  </tr>
</table>

</div>

---

## 🌐 Connect with Me  
📫 Email: [siddhuushakyaa@gmail.com](mailto:siddhuushakyaa@gmail.com)  
💻 GitHub: [@SiddhuShkya](https://github.com/SiddhuShkya)  
🔗 LinkedIn: [Siddhartha Shakya](https://www.linkedin.com/in/siddhartha-shakya-5665a0236/)  
🧑‍💻 LeetCode: [SiddhuShkya](https://leetcode.com/SiddhuShkya)  

---

### 🐍 Contribution Graph  
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/SiddhuShkya/SiddhuShkya/output/snake.svg?palette=github-dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/SiddhuShkya/SiddhuShkya/output/snake.svg?palette=github-light" />
  <img src="https://raw.githubusercontent.com/SiddhuShkya/SiddhuShkya/output/snake.gif" alt="Snake animation showing contributions from 2023–2025" />
</picture>

"""

# Create a Markdown text splitter with 500 character chunks
splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.MARKDOWN,
    chunk_size = 500,
    chunk_overlap = 0
)

chunks = splitter.split_text(md)  # Split the markdown text into chunks

print(len(chunks))  # Print number of chunks
print(f"Total chunks: {len(chunks)}\n")  # Print total chunk info
for i, chunk in enumerate(chunks, 1):  # Loop through chunks
    print(f"--- Chunk {i} ---")  # Print chunk number
    print(chunk)  # Print chunk content
