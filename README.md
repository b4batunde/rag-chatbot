rag-chatbot/
├── data/
│ └── python_guide.txt # Knowledge base
├── chatbot.py # Console chatbot using LangChain
├── streamlit_app.py # Streamlit web interface
├── responses.xlsx # Sample Q&A file
├── requirements.txt # Dependencies
└── README.md # This file


#1. Setup Virtual Environment (macOS/Linux)
python3 -m venv venv
source venv/bin/activate
#2. Install Requirements
pip install -r requirements.txt
#3. Set OpenAI API Key
export OPENAI_API_KEY="sk-..."
#4. Run Console Chatbot
python chatbot.py
#5. Run Streamlit App
streamlit run streamlit_app.py



Citations : https://python.langchain.com/docs/introduction/ | chatGPT(python_guide.txt)