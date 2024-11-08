# Custom NEWS_SEARCH_WORKFLOW for News and Information Retrieval

This project is a custom **NEWS_SEARCH_WORKFLOW** built to retrieve and evaluate relevant, credible news articles based on user queries. The agent leverages OpenAI's language models to intelligently search, evaluate, and return the most pertinent results.

## Key Features

1. **Intelligent Query Processing**: The agent assesses query details, such as keywords and tags, prioritizing high-impact tags based on weight and relevance.
2. **Time-Sensitive Searches**: Automatically adjusts the search timeframe, using recent articles for ongoing events or a broader range for general topics.
3. **Result Evaluation**: Searches are refined and evaluated to ensure only the top matching articles are returned.

## Setup Instructions

### 1. Environment Configuration

Create a `.env` file with your API keys and configuration settings:

```plaintext
OPENAI_API_KEY=your-api-key
TAVILY_API_KEY=your-tavily-api-key (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-langchain-api-key"
LANGCHAIN_PROJECT="your-langchain-project"
```

### 2. Installation
Ensure you have Python 3.9 or later. Then, install the dependencies:

```bash
pip install -r requirements.txt
```

### OR

### install as package

To install this project, clone the repository and install it using the following steps:

**Clone the Repository**:

   ```bash
   git clone https://github.com/username/repo-name.git
   cd repo-name
   pip install -e .
    ```
### 3. Running the Application
After setup, you can run the agent to handle queries:
```python
import asyncio
from graph import run_workflow
from dotenv import load_dotenv
load_dotenv()

# Define the user input
user_input = "Dominion Voting Systems,Defamation Case"

# Run the workflow in an asynchronous context
result = asyncio.run(run_workflow(user_input))

# Print the result
print(result)

```

## Configuration
All configuration settings are managed in configuration.py, allowing you to customize:

- Model: Adjust the language model by specifying a different OpenAI model.
- Search Limits: Modify the maximum search results returned per query.
## Project Structure
- graph.py: Core logic of the NEWS_SEARCH_WORKFLOW, defining the decision-making and reasoning flow.
- tools.py: Defines the tools for searching and retrieving web results, currently using DuckDuckGo.
- utils.py: Utility functions for processing responses and structuring outputs.
- prompts.py: Configurable prompts for guiding the agent’s responses and evaluations.
- state.py: Defines the structure of conversation state and result storage.
## Customization
To expand the agent’s functionality:

1. Add New Tools: Implement custom search or data processing tools in tools.py.
2. Adjust Prompting: Modify prompts.py to fine-tune the agent’s behavior and response style.
3. Extend Evaluation: Enhance result evaluation in state.py to improve filtering or ranking.