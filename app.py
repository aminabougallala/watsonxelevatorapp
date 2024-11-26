import streamlit as st
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from services import get_answer_with_context, log_payload

def main():
    #use config.env file to load environment variables
    deployed_on_oc = True

    if not deployed_on_oc:
        load_dotenv('config.env')

    st.set_page_config(page_title="Elevator Q&A App")
    st.title("Elevator Q&A App")
    st.write("This app answers questions about elevators using an Azure OpenAI 3.5-Turbo LLM.")
    st.write("Payload data is then sent to Watsonx.governance to monitor the output.")


    # Q&A Logic
    if 'client_id' not in st.session_state:
        st.session_state.client_id = ""
    if 'client_secret' not in st.session_state:
        st.session_state.client_secret = ""
    if 'region' not in st.session_state:
        st.session_state.region = ""
    if 'azure_saved' not in st.session_state:
        st.session_state.azure_saved = False
    if 'original_question' not in st.session_state:
        st.session_state.original_question = ""
    if 'azure_response' not in st.session_state:
        st.session_state.azure_response = ""

    if not st.session_state.azure_saved:
        client_id_input = st.text_input("Enter your Azure Client ID:", type="password")
        client_secret_input = st.text_input("Enter your Azure Client Secret:", type="password")
        region_input = st.selectbox("Select your Azure region:", ["", "East US", "Japan East", "West Europe"])
        if st.button("Save Azure Credentials", key="azure_save_button"):
            if client_id_input and client_secret_input and region_input:
                st.session_state.client_id = client_id_input
                st.session_state.client_secret = client_secret_input
                st.session_state.region = region_input
                st.session_state.azure_saved = True
                st.success("Azure Credentials saved successfully. (Click again to hide)")
        elif st.session_state.azure_saved:
            if st.button("Hide me", key="azure_hide_button"):
                st.session_state.azure_saved = False

    input_question = st.text_area("Enter your elevator-related question:", height=100, value=st.session_state.original_question)
    st.session_state.original_question = input_question

    if st.button("Get Answer"):
        if st.session_state.client_id == "" or st.session_state.client_secret == "" or st.session_state.region == "":
            st.error("Please enter your Azure credentials and select a region.")
        elif input_question.strip() == "":
            st.error("Please enter a question about elevators.")
        else:
            # Set up parameters
            azure_params = {
                "client_id": st.session_state.client_id,
                "client_secret": st.session_state.client_secret,
                "region": st.session_state.region
            }
            watsonx_params = {
                "api_key": os.environ.get("IBM_CLOUD_KEY"),
                "ibm_cloud_url": os.environ.get("IBM_CLOUD_URL"),
                "project_id": os.environ.get("IBM_CLOUD_PROJECT_ID")
            }
            milvus_params = {
                "milvus_url": os.environ.get("MILVUS_URL"),
                "milvus_port": os.environ.get("MILVUS_PORT"),
                "ibm_cloud_key": os.environ.get("IBM_CLOUD_KEY")
            }

            use_cpd = os.environ.get("USE_CPD")
            if use_cpd == "True":
                project_id = os.environ.get("CPD_PROJECT_ID")
            else:
                project_id = os.environ.get("IBM_CLOUD_PROJECT_ID")
            wxgov_params = {
                "use_cpd": os.environ.get("USE_CPD"),
                "cpd_url": os.environ.get("CPD_URL"),
                "cpd_username": os.environ.get("CPD_USERNAME"),
                "cpd_password": os.environ.get("CPD_PASSWORD"),
                "ibm_cloud_key": os.environ.get("IBM_CLOUD_KEY"),
                "wos_subscription_id": os.environ.get("WOS_SUBSCRIPTION_ID"),
                "project_id": f"{project_id}"
            }

            # Get the response from OpenAI
            response, context, input_tokens, output_tokens, response_time = get_answer_with_context(input_question, azure_params, watsonx_params, milvus_params)
            st.session_state.azure_response = response
            st.subheader("Azure Response")
            st.write(st.session_state.azure_response)

            # Log the payload in the background
            pl_request, pl_response = log_payload(wxgov_params, input_question, response, context, input_tokens, output_tokens, response_time)
            st.info("OpenScale payload logged successfully.")

            # Display payload data in an expander
            with st.expander("View Payload Data"):
                st.write("**Payload Request:**")
                st.json(pl_request)
                st.write("**Payload Response:**")
                st.json(pl_response)


if __name__ == "__main__":
    main()
