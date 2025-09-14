
# pdf-rag-chat

`pdf-rag-chat` is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and chat with them using a large language model. The application is built with FastAPI and  extract just text from the pdf document.

## Features

*   **PDF Upload:** Users can upload PDF documents through a simple web interface.
*   **Document Processing:** The application uses the `unstructured` library to partition the PDF into text and images. The text is then chunked by title to maintain semantic context.
*   **Vector Store:** The processed chunks are stored in a ChromaDB vector store. The embeddings are generated using Ollama.
*   **Chat Interface:** A simple chat interface allows users to ask questions about the uploaded document.
*   **Background Processing:** The PDF processing and indexing is done in the background to avoid blocking the UI.
*   **Status Polling:** The UI polls the server to check the status of the indexing process.
*   **Overwrite Confirmation:** The user is warned if an existing index will be overwritten.

## Technology Stack

*   **Backend:** FastAPI, Uvicorn
*   **Frontend:** HTML, CSS, JavaScript
*   **Document Processing:** `unstructured`, `langchain`, `pypdf`
*   **Vector Store:** ChromaDB
*   **Embeddings:** Ollama
*   **LLM:** OpenAI (for summarization and chat)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/pdf-rag-chat.git
    cd pdf-rag-chat
    ```

2.  **Install the dependencies:**

    ```bash
    uv sync
    ```

3.  **Set up your environment variables:**

    Create a `.env` file in the root of the project and add your OpenAI API key:

    ```
    OPENAI_API_KEY=your-api-key
    ```

## Usage

1.  **Run the application:**

    ```bash
    uv run main.py
    ```

2.  **Open your browser:**

    Navigate to `http://127.0.0.1:8000`.

3.  **Upload a PDF:**

    *   Click on the "Upload PDF" link in the navigation.
    *   Select a PDF file to upload.
    *   The application will start processing the document in the background.

4.  **Chat with the document:**

    *   Once the indexing is complete, you can go to the "Chat" page.
    *   Ask questions about the document in the chat interface.

## API Endpoints

*   `GET /`: The home page.
*   `GET /upload`: The file upload page.
*   `POST /upload`: The endpoint for uploading a PDF file.
*   `GET /status/{task_id}`: The endpoint to check the status of the indexing process.
*   `GET /index-status`: The endpoint to check if an index is already available.
*   `GET /chat`: The chat page.
*   `POST /chat`: The endpoint for sending a chat message.

## Future Improvements

*   **Support for more file types:** The application could be extended to support other file types like DOCX, TXT, etc.
*   **More advanced chat features:** The chat interface could be improved with features like chat history, user authentication, etc.
*   **Better error handling:** The error handling could be made more robust.
*   **More sophisticated RAG pipeline:** The RAG pipeline could be improved with more advanced techniques like re-ranking, query transformations, etc.
