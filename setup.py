# setup.py
from setuptools import setup, find_packages

setup(
    name="search_workflow",
    version="0.1.0",
    description="Custom Reasoning and Action agent with tool support",
    author="Your Name",
    author_email="your.email@example.com",
    package_dir={"": "src"},  
    packages=find_packages(where="src"),  
    install_requires=[
        "langchain-core>=0.2.14",
        "langchain-openai>=0.1.22",
        "langgraph>=0.2.6",
        "pydantic>=1.10.0",
        "python-dotenv>=1.0.1",
        "duckduckgo-search",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "mypy",
            "ruff",
        ]
    },
    python_requires=">=3.9",
)
