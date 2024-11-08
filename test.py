import asyncio
from search_workflow import run_workflow
from dotenv import load_dotenv
load_dotenv()

# Define the user input
user_input = "language='english' tags=[TagInfo(value='Dominion Voting Systems', tag_type='Topic', category_high_level='Legal Matters', category_low_level='Defamation Case', weight=1.0), TagInfo(value='Fox News', tag_type='Topic', category_high_level='Media', category_low_level='News Network', weight=1.0)]'"

# Run the workflow in an asynchronous context
result = asyncio.run(run_workflow(user_input))

# Print the result
print("Result:", result)
print('#' * 30)
print("Type:", type(result))
print('#' * 30)
print("Type:", type(result[0]))
print('#' * 30)
print("Length:", len(result))
