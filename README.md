# Python LLM Service SSE for Qlik Sense Client Managed

## Server Setup
Ensure you have Python installed.

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

Create a .env file in the root directory with the following contents:
```
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
PORT=50055
```
Replace YOUR_OPENAI_API_KEY with your actual OpenAI API key. The PORT can be modified if needed.

Start the server:
```bash
python __main__.py
```

## Analytic Connection Setup

1. Create a new analytic connection in Qlik Sense Management Console https://your-qlik-server/qmc/analyticconnections -> +Create new
2. Set the connection name to "Python"
3. Set the connection Host and Port of the server where you are running this service
4. Click apply

## Example Qlik Sense app

See `Client Managed LLMs.qvf` for an example of how to use the LLM function in Qlik Sense.

## models.json file
This file contains the models that are available to use in the LLM function. Modify this file to add or remove models which are available to use in the LLM function.

## Function Definitions

### AnalyticConnectionName.LLM
Process data through an LLM model with a custom prompt.

**Parameters:**
- `columns`: String of column names separated by pipes  
  Example: `"Column1|Column2|Column3"`
- `data`: String of data values separated by pipes  
  Example: `"Data1|Data2|Data3"`
- `model`: Model name as defined in models.json  
  Example: `"gpt-4"`
- `prompt`: String containing the prompt text

### AnalyticConnectionName.GetTokens
Calculate token count for given data and model.

**Parameters:**
- `columns`: String of column names separated by pipes  
  Example: `"Column1|Column2|Column3"`
- `data`: String of data values separated by pipes  
  Example: `"Data1|Data2|Data3"`
- `model`: Model name as defined in models.json  
  Example: `"gpt-4"`

### AnalyticConnectionName.LLMStructured
Parse data into a structured format using an LLM.

**Parameters:**
- `data`: String data to be parsed into structured format
- `functionDefinition`: Schema definition in format "name|description|type,name2|description2|type2"  
  Example: `"product|product name|string,price|product price|number,available|stock status|boolean"`
- `model`: Model name as defined in models.json  
  Example: `"gpt-4"`
- `return_type`: Output format ("json" or "string"). json should be used when passing an entire table in the load script.   
  Example: `"json"`

**Notes:**
- Valid types for schema fields include: string, number, integer, boolean, object, array, null
- When return_type is "json", the function returns multiple columns corresponding to each defined field. Only use this when passing an entire table in the load script.
- For the string return type, the function returns the entire JSON response as a string. You can use the JSONGet() to extract the data you need.



    