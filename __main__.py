#! /usr/bin/env python3
import argparse
import json
import logging
import logging.config
import os
import sys
import time
import re
from concurrent import futures
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# Add Generated folder to module path.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, "generated"))

import ServerSideExtension_pb2 as SSE
import grpc
from ssedata import FunctionType
from scripteval import ScriptEval

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Load models configuration
models_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models.json"
)
models = None
with open(models_file) as f:
    models = json.load(f)
valid_models = [model["name"] for model in models]      

def get_tokens(model, data_rows):
    encoding = tiktoken.encoding_for_model(model)
    print(len(encoding.encode(data_rows)))
    return len(encoding.encode(data_rows))

def validate_model(model, valid_models):
    """
    Validates if the model is in the list of valid models.
    Returns (is_valid, error_message) tuple.
    """
    if model not in valid_models:
        error_message = f"Invalid model: {model}. Valid models are: {', '.join(valid_models)}"
        return False, error_message
    return True, None

def validate_token_count(model, token_count, models):
    """
    Validates if the token count is within the model's maximum limit.
    Returns (is_valid, error_message) tuple.
    """
    model_info = next((m for m in models if m["name"] == model), None)
    if model_info and token_count <= model_info["max_tokens"]:
        return True, None
    return (
        False,
        f"Token count {token_count} exceeds maximum limit of {model_info['max_tokens']} for model {model}",
    )

def get_csv_string(columns, data_rows):
    csv_columns = ','.join(f'"{col}"' for col in columns.split('|'))
    csv_rows = []
    for row in data_rows:
        values = row.split('|')
        csv_row = ','.join(f'"{val}"' for val in values)
        csv_rows.append(csv_row)
    return csv_columns + '\n' + '\n'.join(csv_rows)

class ExtensionService(SSE.ConnectorServicer):
    """
    A simple SSE-plugin created for the HelloWorld example.
    """

    def __init__(self, funcdef_file):
        """
        Class initializer.
        :param funcdef_file: a function definition JSON file
        """
        self._function_definitions = funcdef_file
        self.ScriptEval = ScriptEval()


        os.makedirs("logs", exist_ok=True)
        log_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logger.config"
        )
        logging.config.fileConfig(log_file)
        logging.info("Logging enabled")

    @property
    def function_definitions(self):
        """
        :return: json file with function definitions
        """
        return self._function_definitions

    @property
    def functions(self):
        """
        :return: Mapping of function id and implementation
        """
        return {0: "_llm", 1: "_get_tokens"}

    @staticmethod
    def _get_function_id(context):
        """
        Retrieve function id from header.
        :param context: context
        :return: function id
        """
        metadata = dict(context.invocation_metadata())
        header = SSE.FunctionRequestHeader()
        header.ParseFromString(metadata["qlik-functionrequestheader-bin"])

        return header.functionId

    """
    Implementation of added functions.
    """
    
    @staticmethod
    def _get_tokens(request, context):
        global models
        global valid_models
        global llm_client
        global get_tokens
        global get_csv_string
        
        # Initialize variables to store data
        columns = None
        data = None
        model = None
        data_rows = []

        # Process each bundle
        for request_bundle in request:
            # Get first row data if not already set
            if model is None:
                first_row = request_bundle.rows[0]
                # Update order to match functions.json: columns, data, model
                columns = first_row.duals[0].strData
                data = first_row.duals[1].strData
                model = first_row.duals[2].strData
                
            # Collect all data rows
            for row in request_bundle.rows:
                data_rows.append(row.duals[1].strData)

        # Validate model
        is_valid, error_message = validate_model(model, [m["name"] for m in models])
        if not is_valid:
            response_rows = [SSE.Row(duals=[SSE.Dual(strData=error_message)])]
            yield SSE.BundledRows(rows=response_rows)
            return

        # get csv string
        csv_string = get_csv_string(columns, data_rows)
        
        # Get token count
        token_count = get_tokens(model, csv_string)
        
        # Validate token count
        is_valid, error_message = validate_token_count(model, token_count, models)
        if not is_valid:
            response_rows = [SSE.Row(duals=[SSE.Dual(strData=error_message)])]
            yield SSE.BundledRows(rows=response_rows)
            return

        # Create a single row with the token count as a numeric value
        response_rows = [SSE.Row(duals=[SSE.Dual(numData=token_count)])]

        # Return the bundled row
        yield SSE.BundledRows(rows=response_rows)

    @staticmethod   
    def _llm(request, context):
        global models
        global valid_models
        global llm_client
        global get_tokens
        
        # Initialize variables to store data
        columns = None
        data = None
        model = None
        prompt = None
        data_rows = []

        # Process each bundle
        for request_bundle in request:
            # Get first row data if not already set
            if model is None:
                first_row = request_bundle.rows[0]
                # Update order to match functions.json: columns, data, model, prompt
                columns = first_row.duals[0].strData
                data = first_row.duals[1].strData
                model = first_row.duals[2].strData
                prompt = first_row.duals[3].strData

            # Collect all data rows
            for row in request_bundle.rows:
                data_rows.append(row.duals[1].strData)

        # Create the CSV-formatted string using the helper function
        data_string = get_csv_string(columns, data_rows)
        
        # make call to LLM
        completion = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": "You are a helpful assistant. Which should answer questions about the data you received"},
                {
                    "role": "user",
                    "content": f"{prompt}\n\n Data:\n{data_string}",
                },
            ],
        )
        message = completion.choices[0].message.content
        
        # For debugging
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print(f"Columns: {columns}")
        print(f"Data:\n{data_string}")
        print(f"Message:\n{message}")
        
        # Create the dual with the result
        duals = [SSE.Dual(strData=message)]

        # Yield the row data as bundled rows
        yield SSE.BundledRows(rows=[SSE.Row(duals=duals)])

    """
    Implementation of rpc functions.
    """

    def GetCapabilities(self, request, context):
        """
        Get capabilities.
        Note that either request or context is used in the implementation of this method, but still added as
        parameters. The reason is that gRPC always sends both when making a function call and therefore we must include
        them to avoid error messages regarding too many parameters provided from the client.
        :param request: the request, not used in this method.
        :param context: the context, not used in this method.
        :return: the capabilities.
        """
        logging.info("GetCapabilities")
        # Create an instance of the Capabilities grpc message
        # Enable(or disable) script evaluation
        # Set values for pluginIdentifier and pluginVersion
        capabilities = SSE.Capabilities(
            allowScript=True, pluginIdentifier="Sentiment", pluginVersion="v1.1.0"
        )

        # If user defined functions supported, add the definitions to the message
        with open(self.function_definitions) as json_file:
            # Iterate over each function definition and add data to the capabilities grpc message
            for definition in json.load(json_file)["Functions"]:
                function = capabilities.functions.add()
                function.name = definition["Name"]
                function.functionId = definition["Id"]
                function.functionType = definition["Type"]
                function.returnType = definition["ReturnType"]

                # Retrieve name and type of each parameter
                for param_name, param_type in sorted(definition["Params"].items()):
                    function.params.add(name=param_name, dataType=param_type)

                logging.info(
                    "Adding to capabilities: {}({})".format(
                        function.name, [p.name for p in function.params]
                    )
                )

        return capabilities

    def ExecuteFunction(self, request_iterator, context):
        """
        Execute function call.
        :param request_iterator: an iterable sequence of Row.
        :param context: the context.
        :return: an iterable sequence of Row.
        """
        # Retrieve function id
        func_id = self._get_function_id(context)

        # Call corresponding function
        logging.info("ExecuteFunction (functionId: {})".format(func_id))

        return getattr(self, self.functions[func_id])(request_iterator, context)

    def EvaluateScript(self, request, context):
        """
        This plugin provides functionality only for script calls with no parameters and tensor script calls.
        :param request:
        :param context:
        :return:
        """
        # Parse header for script request
        metadata = dict(context.invocation_metadata())
        header = SSE.ScriptRequestHeader()
        header.ParseFromString(metadata["qlik-scriptrequestheader-bin"])

        # Retrieve function type
        func_type = self.ScriptEval.get_func_type(header)

        # Verify function type
        if (func_type == FunctionType.Aggregation) or (
            func_type == FunctionType.Tensor
        ):
            return self.ScriptEval.EvaluateScript(header, request, context, func_type)
        else:
            # This plugin does not support other function types than aggregation  and tensor.
            # Make sure the error handling, including logging, works as intended in the client
            msg = "Function type {} is not supported in this plugin.".format(
                func_type.name
            )
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(msg)
            # Raise error on the plugin-side
            raise grpc.RpcError(grpc.StatusCode.UNIMPLEMENTED, msg)

    """
    Implementation of the Server connecting to gRPC.
    """

    def Serve(self, port, pem_dir):
        """
        Sets up the gRPC Server with insecure connection on port
        :param port: port to listen on.
        :param pem_dir: Directory including certificates
        :return: None
        """
        # Create gRPC server
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        SSE.add_ConnectorServicer_to_server(self, server)

        if pem_dir:
            # Secure connection
            with open(os.path.join(pem_dir, "sse_server_key.pem"), "rb") as f:
                private_key = f.read()
            with open(os.path.join(pem_dir, "sse_server_cert.pem"), "rb") as f:
                cert_chain = f.read()
            with open(os.path.join(pem_dir, "root_cert.pem"), "rb") as f:
                root_cert = f.read()
            credentials = grpc.ssl_server_credentials(
                [(private_key, cert_chain)], root_cert, True
            )
            server.add_secure_port("[::]:{}".format(port), credentials)
            logging.info(
                "*** Running server in secure mode on port: {} ***".format(port)
            )
        else:
            # Insecure connection
            server.add_insecure_port("[::]:{}".format(port))
            logging.info(
                "*** Running server in insecure mode on port: {} ***".format(port)
            )

        # Start gRPC server
        server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", nargs="?", default="50055")
    parser.add_argument("--pem_dir", nargs="?")
    parser.add_argument("--definition_file", nargs="?", default="functions.json")
    args = parser.parse_args()

    # need to locate the file when script is called from outside it's location dir.
    def_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.definition_file
    )

    calc = ExtensionService(def_file)
    calc.Serve(args.port, args.pem_dir)
