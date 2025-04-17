import requests
import json

def get_transformer_response(url, prompt):
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if result and "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            if message and "content" in message:
                print("\nModel API response: ", message["content"])
                return message["content"]
        return "Error occurred when parsing response"
    
    except requests.exceptions.RequestException as e:
        return f"Request error: {str(e)}"
    except json.JSONDecodeError:
        return "Error occurred when parsing response"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    url = "" # Placeholder for the API URL
    response = get_transformer_response(url, "Hi there")
    print(response)
