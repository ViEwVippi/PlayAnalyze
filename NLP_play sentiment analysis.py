import streamlit as st
from openai import OpenAI
import pandas as pd
from io import StringIO
from PyPDF2 import PdfReader
import json

# Sidebar for API Key
st.sidebar.title("Settings")
st.sidebar.text("Provide your OpenAI API Key")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")


# App Title and Description
st.title("Play's Sentiment Analysis")
st.markdown(
    """<p>
    <strong>NLP Application for Sentiment Analysis on Plays</strong><br>
    This application using the ChatGPT 3.5 Turbo model.<\n>
    This program is designed to assist actors or directors in interpreting the emotional roles of characters in various scenes of a play and recording the statistics in a table <\n>
    , which contains Dialogue Analyze and Emotion statistics tables for the reasults.<\n>
    &nbsp;&nbsp;&nbsp;&nbsp;- The scritpt shouldn't take long more than 5 minutes/play (normaly between 2 - 3 pages) which can help you to understand each beat(bit) in acting.<\n>
    &nbsp;&nbsp;&nbsp;&nbsp;- If the file of the play is too large, it can occur error. Please split the file.<\n>
    &nbsp;&nbsp;&nbsp;&nbsp;- You may upload your play in PDF, or TXT format. (PDF must be 'text' pdf file)<\n>
    &nbsp;&nbsp;&nbsp;&nbsp;- Error can rarely occur. (due to Api and network problem) Please try again.<\n>
    &nbsp;&nbsp;&nbsp;&nbsp;- You may specify a context to help ChatGPT's analyzing better (e.g., a specific event, a specify characteristic).<\n>
    &nbsp;&nbsp;&nbsp;&nbsp;- You may download the result in an Excel file after it has been analyzed.<\n>
    </p>""",
    unsafe_allow_html=True,
)

# Input for Play File
st.subheader("Upload Play")
st.text("Please upload your play in PDF, or TXT format.")
file = st.file_uploader("Choose a file", type=["txt", "pdf"])

# Input for Specific Section
st.subheader("Context of the play")
st.text(
    "You may describe the context of the play to help ChatGPT analyze sentiment."
)
context = st.text_input("context", "")

if not file:
    st.warning("Please upload a file to proceed.")
    st.stop()

# Read and Process File
def read_file(file):
    if file is None:
        return ""
    if file.type == "text/plain":
        # If it's a TXT file, return the text directly
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        # If it's a PDF, convert it to text first
        return convert_pdf_to_txt(file)
    else:
        return ""

# Function to convert PDF to TXT
def convert_pdf_to_txt(file):
    """
    Converts a PDF file to plain text.
    
    Parameters:
    - file: The uploaded PDF file
    
    Returns:
    - str: The text content of the PDF
    """
    try:
        reader = PdfReader(file)
        extracted_text = "\n".join([page.extract_text() for page in reader.pages])
        return extracted_text
    except Exception as e:
        st.error(f"Error while converting PDF to TXT: {e}")
        return ""

# Read and process the file
data = read_file(file)

if not data:
    st.error("Unable to read the uploaded file. Please try another file.")
    st.stop()


prompt = """
You are a helpful assistant for analyzing plays. Your task is to analyze the content of the provided play script and output the results in JSON format. If the file is PDF read the whole text 3 times. Make sure Output contains all sentences of dialogue.

1. Analyze all lines in the play and create the following outputs:
   - Dialogue Analysis Table: Each entry should include:
       - "speaker": The name of the character speaking in the play. Leave blank for narration or descriptions.
       - "line": The specific dialogue or text.
       - "sentiment": The sentiment of the line (e.g., positive, negative, neutral).
       - "emotion": Possible emotions associated with the speaker. (all sentences must have emotion given)
       - "reason": The reason why the speaker might be saying this line based on the play's context.

   - Emotion Statistics Table: Each entry should include:
       - "emotion": The name of each identified emotion.
       - "count": The number of times this emotion appears.
       - "description": When and where this emotion commonly appears in the play.

2. Output the result in the following JSON format:
```json
{
    "dialogue_table": [
        {
            "speaker": "Speaker Name",
            "line": "The line spoken by the character.",
            "sentiment": "Sentiment analysis of the line.",
            "emotion": ["Emotion 1", "Emotion 2"],
            "reason": "The reason behind the dialogue."
        },
        ...
    ],
    "emotion_statistics": [
        {
            "emotion": "Emotion Name",
            "count": Number of occurrences,
            "description": "Description of when and where the emotion occurs."
        },
        ...
    ]
}
"""

# Call OpenAI API for Sentiment Analysis
def analyze_sentiments(data_text, con_text, apikey):
    # Set the API key dynamically
    try:
        client = OpenAI(
            api_key= apikey,
            )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt },
                {"role": "user", "content": f'This is the context of the play:\n{con_text}. (If the context given is none or cannot be understand, ignore it)\n This is the play: \n {data_text}'},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API error: {e}")
    
# Perform Sentiment Analysis
if api_key and file:
    if st.button("Analyze"):
        with st.spinner("Analyzing sentiments..."):
            try:
                analysis_result = analyze_sentiments(data, context, api_key)  # Pass the API key dynamically
                result_dict = json.loads(analysis_result)
                dialogue_table = result_dict["dialogue_table"]
                emotion_statistics = result_dict["emotion_statistics"]
                dialogue_df = pd.DataFrame(dialogue_table)
                emotion_stats_df = pd.DataFrame(emotion_statistics)
                st.subheader("Dialogue Analysis")
                st.dataframe(dialogue_df)
                st.subheader("Emotion Statistics")
                st.dataframe(emotion_stats_df)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please provide both the API Key and upload a file to enable the Analyze button.")
