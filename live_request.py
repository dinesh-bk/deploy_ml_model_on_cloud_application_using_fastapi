import requests
import json

# Define the API endpoint
url = "https://ml-fastapi-app2-3d41f3580392.herokuapp.com/predict"

data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

try:
    # Make the request to the API
    response = requests.post(url, json=data)
    
    # Check if the response status code is OK (200)
    if response.status_code == 200:
        # Parse the JSON response
        response_data = response.json()
        # Extract the prediction value
        prediction = response_data.get("prediction", "No prediction available")
        # Print the status code and prediction
        print("Status Code:", response.status_code)
        print("Prediction:", prediction)
    else:
        # Print an error message if the response is not OK
        print("Error: Received a non-200 status code")
        print("Status Code:", response.status_code)
        print("Response:", response.text)

except requests.exceptions.RequestException as e:
    # Handle exceptions related to the request
    print("Error: An error occurred while making the request.")
    print(e)
