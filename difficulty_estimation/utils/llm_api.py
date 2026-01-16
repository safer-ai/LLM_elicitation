import os
import anthropic
import time

def get_llm_response(requests: list) -> str:
    """
    Initializes the Anthropic client and gets a response from the specified model.

    Args:
        prompt: The prompt to send to the LLM.
        api_base_url: The base URL for the LLM API endpoint.
        model_name: The name of the model to use.

    Returns:
        The response from the LLM.
    """
    # It is recommended to set the API key as an environment variable
    # e.g., export ANTHROPIC_API_KEY='your-api-key'
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(
        api_key=api_key,
    )

    try:
        batch_params = client.messages.batches.create(
            requests=requests
        )
        print(f"Batch created with ID: {batch_params.id}")
        return batch_params.id, client
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    
    
def await_batch(batch_id: str, client):
    batch_results = {}
    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        print(f"Batch {batch_id} is currently: {message_batch.processing_status}")

        if message_batch.processing_status == "ended":
            print("Batch processing complete!")
            for result_entry in client.messages.batches.results(batch_id):
                if result_entry.result.type == "succeeded":
                    print(f"Request '{result_entry.custom_id}' succeeded:")
                    print(f"  Message ID: {result_entry.result.message.id}")
                    batch_results[result_entry.custom_id] = result_entry.result.message.content[0].text
                elif result_entry.result.type == "failed":
                    print(f"Request '{result_entry.custom_id}' failed:")
                    print(f"  Error: {result_entry.result.error.message}")
            break
        elif message_batch.processing_status in ["failed", "canceled", "expired"]:
            print(f"Batch processing ended with status: {message_batch.processing_status}")
            break
        time.sleep(60) 
    return batch_results

def await_batch_list(batch_id: str, client):
    batch_results = {}
    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        print(f"Batch {batch_id} is currently: {message_batch.processing_status}")

        if message_batch.processing_status == "ended":
            print("Batch processing complete!")
            for result_entry in client.messages.batches.results(batch_id):
                if result_entry.result.type == "succeeded":
                    print(f"Request '{result_entry.custom_id}' succeeded:")
                    print(f"  Message ID: {result_entry.result.message.id}")
                    res = ""
                    for block in result_entry.result.message.content:
                        additional_entries = {}
                        if block.type == "thinking":
                            additional_entries["thinking"] = block.thinking
                        else:
                            batch_results[result_entry.custom_id] = dict(
                                text=block.text,
                                **additional_entries
                            )
                elif result_entry.result.type == "failed":
                    print(f"Request '{result_entry.custom_id}' failed:")
                    print(f"  Error: {result_entry.result.error.message}")
            break
        elif message_batch.processing_status in ["failed", "canceled", "expired"]:
            print(f"Batch processing ended with status: {message_batch.processing_status}")
            break
        time.sleep(60) 
    return batch_results
