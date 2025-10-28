# Quick Start Guide

This guide will help you get LinguaStory up and running quickly.

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd coloring-book-agent
   ```

2. **Set up a Python environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a .env file**
   ```bash
   touch .env
   ```

5. **Add your API keys to the .env file**
   ```
   REPLICATE_API_TOKEN=your_replicate_api_key
   TAVUS_API_KEY=your_tavus_api_key
   ```

## Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at http://localhost:8501.

## Using the Application

1. **Create a Story Book**
   - Go to the "Create Book" tab
   - Enter the child's name
   - Select the language and character style
   - Click "Generate Book"
   - Wait for the story and illustrations to be created

2. **Interactive Conversation**
   - Go to the "Talk with the Character" tab
   - If you've already created a story, it will use those details
   - Otherwise, configure a new character and language
   - Click "Create Conversation"
   - Allow camera and microphone permissions when prompted
   - Start speaking with the AI character

## Troubleshooting

- If you encounter API rate limits, wait a minute and try again
- Make sure your API keys are correctly set in the .env file
- Check that all dependencies are installed
- For Tavus conversation issues, try refreshing the page

## Need Help?

Consult the full documentation in the README.md file or submit an issue on GitHub.
