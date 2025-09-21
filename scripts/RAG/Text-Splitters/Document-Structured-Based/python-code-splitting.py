from langchain.text_splitter import RecursiveCharacterTextSplitter, Language  # Import text splitter for Python code

code = """
class Student:  # Define Student class
    def __init__(self, name, age, grade):  # Constructor to initialize attributes
        self.name = name  # Store student name
        self.age = age  # Store student age
        self.grade = grade  # Store student grade (float)

    def get_details(self):  # Method to return name
        return self.name"

    def is_passing(self):  # Method to check if grade is passing
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)  # Create a Student object
print(student1.get_details())  # Print student name

if student1.is_passing():  # Check if student is passing
    print("The student is passing.")  # Print passing message
else:
    print("The student is not passing.")  # Print not passing message
"""

splitter = RecursiveCharacterTextSplitter.from_language(  # Create text splitter for Python
    language = Language.PYTHON,  # Set language as Python
    chunk_size = 300,  # Maximum chunk size
    chunk_overlap = 0  # No overlap between chunks
)

chunks = splitter.split_text(code)  # Split the code into chunks
print(len(chunks))  # Print number of chunks

print(f"Total chunks: {len(chunks)}\n")  # Print total chunks
for i, chunk in enumerate(chunks, 1):  # Loop through chunks
    print(f"--- Chunk {i} ---")  # Print chunk number
    print(chunk)  # Print chunk content
