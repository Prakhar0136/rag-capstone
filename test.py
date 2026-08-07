import requests

# 1. Send a GET request to the API URL
url = "https://official-joke-api.appspot.com/random_joke"
response = requests.get(url)

# 2. Convert the response into a JSON dictionary
data = response.json()

# 3. Print the raw JSON data
print("Raw JSON Response:")
print(data)
print("-" * 20)

# 4. Extract specific values using their 'keys'
print("Extracted Data:")
print(f"Setup: {data['setup']}")
print(f"Punchline: {data['punchline']}")