# OpenManus-Gemini (grego) ♊

<p>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.12-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
</p>

A fork of the FoundationAgents/OpenManus project, modified to use Google's Gemini models as the primary LLM. Integrating `litellm` allows for easy switching between AI providers like OpenAI or Anthropic. It retains all core agent capabilities—such as web searching, file management, and code execution—powered by the Gemini API.

## About This Project

This project adapts the original OpenManus autonomous agent to leverage the power of Google's Gemini models. The core logic has been refactored to use the `litellm` library, providing a unified and flexible interface for connecting to various Large Language Model providers. This allows you to easily switch between models from Google, OpenAI, Anthropic, Ollama, and more with a simple configuration change.

The agent is capable of complex, multi-step tasks that involve:

* **Web Research:** Searching the internet to find information.
* **Code Generation:** Writing Python scripts to perform tasks.
* **File System Operations:** Creating, reading, and modifying files and directories.
* **Task Automation:** Chaining these abilities together to achieve high-level goals.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management.
* Python 3.12
* A Google AI API key.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SuperAndy777/grego.git](https://github.com/SuperAndy777/grego.git)
    cd grego
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n open_manus python=3.12
    conda activate open_manus
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright browsers:** The agent uses Playwright for web browsing.
    ```bash
    playwright install
    ```

### Configuration

You must configure the application with your API key before running it.

1.  **Create your configuration file:** Copy the example configuration file. This project uses a Gemini-specific template.
    ```bash
    cp config/config.gemini.example.toml config/config.toml
    ```

2.  **Add your API key:** Open `config/config.toml` with a text editor:
    ```bash
    nano config/config.toml
    ```
    Inside the file, find the line `api_key = "YOUR_GOOGLE_API_KEY"` under both the `[llm.default]` and `[llm.google]` sections and replace the placeholder with your actual Google AI API key.

    **Note:** The `config.toml` file is included in `.gitignore` to prevent you from accidentally committing your secret API key.

## Usage

Once the installation and configuration are complete, you can run the agent.

1.  **Activate the Conda environment (if you haven't already):**
    ```bash
    conda activate open_manus
    ```

2.  **Run the main script:**
    ```bash
    python main.py
    ```

3.  **Provide a prompt:** When prompted, give the agent a specific, detailed task. A good prompt should have a clear goal and define what "done" looks like.

    > **Example Prompt:**
    > "Find the top 3 trending topics on Google Trends for the United States. For each topic, perform a web search to find a recent news headline. Compile the topics and their corresponding headlines into a single report named 'trends_report.txt'."

The agent will then begin its execution loop. Any files it creates will be saved in the `workspace/` directory.

## Acknowledgements

* This project is a fork of the original [OpenManus](https://github.com/FoundationAgents/OpenManus) project.
* Special thanks to the [LiteLLM](https://github.com/BerriAI/litellm) team for creating a fantastic multi-provider LLM library.

