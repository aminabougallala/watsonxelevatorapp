import os
from azure.identity import ClientSecretCredential, get_bearer_token_provider
from datetime import datetime
from openai import AzureOpenAI
from ibm_watsonx_ai import APIClient as WatsonXClient
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watson_openscale import APIClient
from ibm_watson_openscale.data_sets import DataSetTypes, TargetTypes
from ibm_watson_openscale.supporting_classes.payload_record import PayloadRecord
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator, CloudPakForDataAuthenticator
import requests
from pymilvus import(
    Milvus,
    IndexType,
    Status,
    connections,
    FieldSchema,
    DataType,
    Collection,
    CollectionSchema,
)


class AzureOpenAIService:
    def __init__(self, client_id, client_secret, region):
        self.tenant_id = "4e7730a0-17bb-4dfa-8dad-7c54d3e761b7"  # Fixed Azure Tenant ID
        self.client_id = client_id
        self.client_secret = client_secret
        self.region = region
        self.api_version = "2024-02-15-preview"
        self.azure_endpoint, self.deployment_name = self._get_deployment_details()
        self.client = self._connect_to_openai()

    def _get_deployment_details(self):
        if self.region == "East US":
            return "https://azureml-openai-americas-1.openai.azure.com/", "tz-gpt-35-turbo-americas-1"
        elif self.region == "Japan East":
            return "https://azureml-openai-apac-1.openai.azure.com/", "tz-gpt-35-turbo-apac-1"
        elif self.region == "West Europe":
            return "https://azureml-openai-emea-1.openai.azure.com/", "tz-gpt-35-turbo-emea-1"

    def _connect_to_openai(self):
        credential = ClientSecretCredential(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        return AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_ad_token_provider=token_provider,
        )

    def get_response(self, context, input_question):
        content = f"Answer the below question from the given context only and do not use the knowledge outside the context. If you do not know the answer, respond I do not know.\n\nContext: {context[0]} {context[1]} {context[2]} {context[3]}\nQuestion: {input_question}\nAnswer:"
        now = datetime.now()
        now = datetime.timestamp(now)
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        output_tokens = response.usage.completion_tokens
        input_tokens = response.usage.prompt_tokens
        response_time = int(round(response.created - now, 3) * 1000)
        response = response.choices[0].message.content
        return response, input_tokens, output_tokens, response_time

class WatsonXAIService:
    def __init__(self, api_key, ibm_cloud_url, project_id):
        self.credentials = {
            "url": ibm_cloud_url,
            "apikey": api_key
        }
        self.project_id = project_id

    def embed_query(self, query):
        embedding = Embeddings(
            model_id="sentence-transformers/all-minilm-l12-v2",
            credentials=self.credentials,
            project_id=self.project_id
        )
        return embedding.embed_query(text=query)

class MilvusService:
    def __init__(self, milvus_url, milvus_port, ibm_cloud_key):
        connections.connect(alias="default", 
                    host=milvus_url, 
                    port=milvus_port, 
                    user='ibmlhapikey', 
                    password=ibm_cloud_key, 
                    secure=True)

    def query(self, query_embeddings, num_results=4):
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 5}
        }
        basic_collection = Collection('wiki_elevator_articles')
        results = basic_collection.search(
            data=[query_embeddings],
            anns_field="vector",
            param=search_params,
            limit=num_results,
            expr=None,
            output_fields=['article_text'],
        )
        self.context={}
        self.context[0]=results[0][0].entity.get('article_text')
        self.context[1]=results[0][1].entity.get('article_text')
        self.context[2]=results[0][2].entity.get('article_text')
        self.context[3]=results[0][3].entity.get('article_text')

        return self.context


def get_answer_with_context(input_question, azure_params, watsonx_params, milvus_params):
    watsonx_service = WatsonXAIService(**watsonx_params)
    milvus_service = MilvusService(**milvus_params)
    azure_service = AzureOpenAIService(**azure_params)

    query_embeddings = watsonx_service.embed_query(input_question)
    context = milvus_service.query(query_embeddings)
    response, input_tokens, output_tokens, response_time = azure_service.get_response(context, input_question)
    return response, context, input_tokens, output_tokens, response_time

def log_payload(wxgov_params, input_question, response, context, input_tokens, output_tokens, response_time):
    use_cpd = wxgov_params['use_cpd']
    if use_cpd == "True":
        authenticator = CloudPakForDataAuthenticator(
            url=wxgov_params['cpd_url'],
            username=wxgov_params['cpd_username'],
            password=wxgov_params['cpd_password'],
            disable_ssl_verification=True
        )
        wos_client = APIClient(
            service_url=wxgov_params["cpd_url"],
            authenticator=authenticator,
            service_instance_id=None
        )
    else:
        authenticator = IAMAuthenticator(
            apikey=wxgov_params["ibm_cloud_key"],
            url="https://iam.cloud.ibm.com"
        )
        wos_client = APIClient(
            authenticator=authenticator,
            service_url="https://aiopenscale.cloud.ibm.com"
        )
    SUBSCRIPTION_ID = wxgov_params['wos_subscription_id']
    PROJECT_ID = wxgov_params['project_id']
    payload_logging_data_set_id = wos_client.data_sets.list(
        type=DataSetTypes.PAYLOAD_LOGGING,
        target_target_id=SUBSCRIPTION_ID,
        target_target_type=TargetTypes.SUBSCRIPTION,
        space_id=PROJECT_ID
    ).result.data_sets[0].metadata.id

    REQUEST_DATA = {
        "parameters": {
            "template_variables": {
                "question": f"{input_question}",
                "context1": f"{context[0]}",
                "context2": f"{context[1]}",
                "context3": f"{context[2]}",
                "context4": f"{context[3]}",
            }
        },
        "project_id": PROJECT_ID
        }
    RESPONSE_DATA = {
        "results": [
            {
                "generated_text": f"{response}",
                "input_token_count": input_tokens,
                "generated_token_count": output_tokens
            }
        ]
    }

    body = PayloadRecord(request=REQUEST_DATA, response=RESPONSE_DATA, response_time=response_time)

    wos_client.data_sets.store_records(
        data_set_id=payload_logging_data_set_id,
        request_body=[body]
    )

    return REQUEST_DATA, RESPONSE_DATA