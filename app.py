from ibm_watson import DiscoveryV2
from ibm_watson.discovery_v2 import QueryLargePassages
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_cloud_sdk_core import IAMTokenManager
from ibm_watson_machine_learning.foundation_models import Model
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
cors = CORS(app)

apikey = os.getenv("APIKEY", None)
discovery_proj_id = os.getenv("DISCOVERY_PROJECTID", None)
service_url = os.getenv("SERVICE_URL", None)
watsonx_proj_id  = os.getenv("WATSONX_PROJECTID", None)

api_key = apikey 
discovery_project_id = discovery_proj_id
service_url = service_url
watsonx_project_id = watsonx_proj_id 

# Create IBM IAM authenticator object using your IBM Cloud API Key
authenticator = IAMAuthenticator(api_key)

# Create an Watson Discovery object using the authenticator object previously created
discovery = DiscoveryV2(
    version='2020-08-30',
    authenticator=authenticator
)

# Set the Watson Discovery service url within our discovery object
discovery.set_service_url(service_url)



def augmenting( template_in, context_in, query_in ):
    return template_in % (  context_in, query_in )

def generate_res( model_in, augmented_prompt_in ):
    
    generated_response = model_in.generate( augmented_prompt_in, guardrails=False)
    #print(generated_response)

    if ( "results" in generated_response ) \
       and ( len( generated_response["results"] ) > 0 ) \
       and ( "generated_text" in generated_response["results"][0] ):
        return generated_response["results"][0]["generated_text"]
    else:
        print( "The model failed to generate an answer" )
        print( "\nDebug info:\n" + json.dumps( generated_response, indent=3 ) )
        return ""
    
def query_discovery(question):
    passages = {
    "enabled": True, 
    "per_document": True, 
    "find_answers": True,
    "max_per_document": 1, 
    "characters": 500
   }
    query_large_passages_model = QueryLargePassages.from_dict(passages)
    
    return discovery.query(
          project_id=discovery_project_id,
          natural_language_query=question,
          passages=query_large_passages_model,
          count=3
      ).get_result()



def handle_wx_wd(question):
    prompt_template = """
Article:
###
%s
###

[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Be brief in your answers. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.\n<</SYS>>\n\nGenerate the next agent response by answering the question. You are provided several documents with titles so find answer from those documents and do not reference documents in answer. If you cannot base your answer on the given documents, please state that you do not have an answer. Keep answer under 150 words and in paragraph format

Question: %s
Answer:
"""
    discovery_json = query_discovery(question)
    disc_results = []
    combined_disc_results = []
    for doc_index in range(len(discovery_json["results"])):
        for j in range(len(discovery_json["results"][doc_index])):
            passages = discovery_json["results"][doc_index]["document_passages"]
            disc_results = []
            for item in passages:
                item = item["passage_text"].replace("<em>","")
                item = item.replace("</em>", "")
                disc_results.append(item)
        combined_disc_results.append("\n".join(disc_results))

    # LLM that we want to use with watsonx.ai
    model_id="google/flan-ul2"
    #"ibm-mistralai/merlinite-7b"
    #"meta-llama/llama-3-70b-instruct" 
    #"google/flan-ul2"


    endpoint= "https://us-south.ml.cloud.ibm.com"
    access_token = ''

    try:
        access_token = IAMTokenManager(
            apikey = api_key,
            url = "https://iam.cloud.ibm.com/identity/token"
        ).get_token()
    except:
        return('Issue obtaining access token. Check variables?')

    credentials = { 
        "url"    : endpoint, 
        "token" : access_token
    }

    # watsonx.ai tuning parameters
    gen_params = {
        "DECODING_METHOD" : "greedy",
        "MAX_NEW_TOKENS" : 100,
        "MIN_NEW_TOKENS" : 1,
        # "STREAM" : False,
        # "TEMPERATURE" : 0,
        # "TOP_K" : 50,
        # "TOP_P" : 1,
        # "RANDOM_SEED" : 10
    }

    model = Model( model_id, credentials, gen_params, watsonx_project_id )

    augmented_prompt = augmenting( prompt_template, combined_disc_results, question )
    output = generate_res( model, augmented_prompt )
    #if not re.match( r"\S+", output ):
        #print( "The model failed to generate an answer")
    #print( "\nAnswer:\n" + output )
    return output
    # print(output)






@app.route('/askwx', methods=['POST'])
def askwx():
    try:

        json_data = request.get_json()
        user_question = json_data["question"]

        result = handle_wx_wd(user_question)  
        return {"answer": result}, 200
    
    except Exception as e:
        return jsonify({"Error": str(e)}), 400

if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 8080, debug = False)