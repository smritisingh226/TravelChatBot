import re
from flask import Flask, render_template, request, jsonify
import random 
import json 
import pickle 
import numpy as np 
import nltk 
from keras.models import load_model 
from nltk.stem import WordNetLemmatizer
import requests, json

top_100_countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
    "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica",
    "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt",
    "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon",
    "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
    "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, North", "Korea, South", "Kosovo",
    "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania",
    "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius",
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway", "Oman",
    "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
    "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino",
    "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
    "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey",
    "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu",
    "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
]

country_currency = {
    'United States': 'USD',
    'China': 'CNY',
    'Japan': 'JPY',
    'Germany': 'EUR',
    'India': 'INR',
    'United Kingdom': 'GBP',
    'France': 'EUR',
    'Brazil': 'BRL',
    'Italy': 'EUR',
    'Canada': 'CAD',
    'South Korea': 'KRW',
    'Russia': 'RUB',
    'Australia': 'AUD',
    'Spain': 'EUR',
    'Mexico': 'MXN',
    'Indonesia': 'IDR',
    'Netherlands': 'EUR',
    'Turkey': 'TRY',
    'Saudi Arabia': 'SAR',
    'Switzerland': 'CHF',
    'Taiwan': 'TWD',
    'Poland': 'PLN',
    'Sweden': 'SEK',
    'Belgium': 'EUR',
    'Thailand': 'THB',
    'Iran': 'IRR',
    'Austria': 'EUR',
    'Norway': 'NOK',
    'United Arab Emirates': 'AED',
    'Nigeria': 'NGN',
    'Argentina': 'ARS',
    'South Africa': 'ZAR',
    'Israel': 'ILS',
    'Ireland': 'EUR',
    'Singapore': 'SGD',
    'Hong Kong': 'HKD',
    'Malaysia': 'MYR',
    'Philippines': 'PHP',
    'Pakistan': 'PKR',
    'Colombia': 'COP',
    'Chile': 'CLP',
    'Bangladesh': 'BDT',
    'Egypt': 'EGP',
    'Vietnam': 'VND',
    'Czech Republic': 'CZK',
    'Portugal': 'EUR',
    'Peru': 'PEN',
    'Greece': 'EUR',
    'New Zealand': 'NZD',
    'Algeria': 'DZD',
    'Romania': 'RON',
    'Iraq': 'IQD',
    'Kazakhstan': 'KZT',
    'Qatar': 'QAR',
    'Hungary': 'HUF',
    'Kuwait': 'KWD',
    'Ukraine': 'UAH',
    'Morocco': 'MAD',
    'Oman': 'OMR',
    'Ecuador': 'USD',
    'Belarus': 'BYN',
    'Slovakia': 'EUR',
    'Sri Lanka': 'LKR',
    'Ethiopia': 'ETB',
    'Dominican Republic': 'DOP',
    'Kenya': 'KES',
    'Guatemala': 'GTQ',
    'Myanmar': 'MMK',
    'Panama': 'PAB',
    'Costa Rica': 'CRC',
    'Uruguay': 'UYU',
    'Lebanon': 'LBP',
    'Croatia': 'HRK',
    'Lithuania': 'EUR',
    'Tanzania': 'TZS',
    'Syria': 'SYP',
    'Jordan': 'JOD',
    'Latvia': 'EUR',
    'Uzbekistan': 'UZS',
    'Ghana': 'GHS',
    'Bulgaria': 'BGN',
    'Serbia': 'RSD',
    'Libya': 'LYD',
    'Yemen': 'YER',
    'Nepal': 'NPR',
    'Bolivia': 'BOB',
    'Cameroon': 'XAF',
    'Cote d\'Ivoire': 'XOF',
    'El Salvador': 'USD',
    'Honduras': 'HNL',
    'Paraguay': 'PYG',
    'Uganda': 'UGX',
    'Zambia': 'ZMW',
    'Trinidad and Tobago': 'TTD',
    'Senegal': 'XOF',
    'Zimbabwe': 'ZWL',
    'Malawi': 'MWK',
    'Namibia': 'NAD',
    'Mozambique': 'MZN',
    'Nicaragua': 'NIO',
    'Botswana': 'BWP',
    'Madagascar': 'MGA',
    'Benin': 'XOF',
    'Rwanda': 'RWF',
    'Niger': 'XOF',
    'Burkina Faso': 'XOF',
    'Mali': 'XOF',
    'Haiti': 'HTG',
    'Chad': 'XAF',
    'Tunisia': 'TND',
    'Bahrain': 'BHD',
    'Cyprus': 'EUR',
    'Republic of the Congo': 'XAF',
    'Jamaica': 'JMD',
    'Mauritius': 'MUR',
    'Albania': 'ALL',
    'Macedonia': 'MKD',
    'Armenia': 'AMD',
    'Botswana': 'BWP',
    'Gambia': 'GMD',
    'Lesotho': 'LSL',
    'Kyrgyzstan': 'KGS',
    'Swaziland': 'SZL',
    'Timor-Leste': 'USD',
    'Guinea': 'GNF',
    'Georgia': 'GEL',
    'Tajikistan': 'TJS',
    'Bhutan': 'BTN',
    'Brunei': 'BND',
    'Fiji': 'FJD',
    'Suriname': 'SRD',
    'Eritrea': 'ERN',
    'Guyana': 'GYD',
    'Mauritania': 'MRU',
    'Maldives': 'MVR',
    'Montenegro': 'EUR',
    'Cape Verde': 'CVE',
    'Sierra Leone': 'SLL',
    'Seychelles': 'SCR',
    'Liechtenstein': 'CHF',
    'San Marino': 'EUR',
    'Solomon Islands': 'SBD',
    'Palau': 'USD',
    'Kiribati': 'AUD',
    'Marshall Islands': 'USD',
    'Tuvalu': 'AUD',
    'Nauru': 'AUD',
    'Tonga': 'TOP',
    'Micronesia': 'USD',
    'Vanuatu': 'VUV',
    'Saint Kitts and Nevis': 'XCD',
    'Saint Vincent and the Grenadines': 'XCD',
    'Grenada': 'XCD',
    'Samoa': 'WST',
    'Dominica': 'XCD',
    'Saint Lucia': 'XCD',
    'Antigua and Barbuda': 'XCD',
    'Barbados': 'BBD',
    'Belize': 'BZD',
    'Sao Tome and Principe': 'STN',
    'Comoros': 'KMF',
    'Saint Helena': 'SHP',
    'Saint Pierre and Miquelon': 'EUR',
    'Montserrat': 'XCD',
    'Falkland Islands': 'FKP',
    'Niue': 'NZD',
    'Cook Islands': 'NZD',
    'Tokelau': 'NZD',
    'Norfolk Island': 'AUD',
    'Christmas Island': 'AUD',
    'Cocos (Keeling) Islands': 'AUD',
    'Pitcairn Islands': 'NZD',
}


app = Flask(__name__)

# Load the chatbot model and data
lemmatizer = WordNetLemmatizer() 
intents = json.loads(open("intense.json").read()) 
words = pickle.load(open('words.pkl', 'rb')) 
classes = pickle.load(open('classes.pkl', 'rb')) 
model = load_model('chatbotmodel.h5') 

def clean_up_sentences(sentence): 
    sentence_words = nltk.word_tokenize(sentence) 
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] 
    return sentence_words 

def bag_of_words(sentence): 
    sentence_words = clean_up_sentences(sentence) 
    bag = [0]*len(words) 
    for w in sentence_words: 
        for i, word in enumerate(words): 
            if word == w: 
                bag[i] = 1
    return np.array(bag) 

def predict_class(sentence): 
    bow = bag_of_words(sentence) 
    res = model.predict(np.array([bow]))[0] 
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_list = [] 
    for r in results: 
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])}) 
    return return_list 

def get_response(intents_list, intents_json, user_message):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""

    if tag == "flight_booking":
        data = user_message
        print(data)
    if tag == "get_country_currency":
        country = extract_city(user_message)
        data = get_currency(country)
        for i in list_of_intents:
            if i['tag'] == tag:
                print(data)
                result = random.choice(i['responses'])
                result += data
                return result
    if tag == "weather_information":
        # Extract the city from the user's message
        city = extract_city(user_message)
        if city:
            result = weather(city)
        else:
            result = "Sorry, I couldn't identify the Country."

    else:
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                print("I cam here")
                break
    return result

def flight_booking(city1, city2):
    # API endpoint URL
    url = 'https://api.tequila.kiwi.com/v2/search'

    IATAcodeOne = getIata(city1)
    IATAcodeTwo = getIata(city2)

    # Headers
    headers = {
        'apikey': 'DJMPv-1doxhjAkXnuxRh5sQndz9-Kyfu'
    }

    # Query parameters
    # params = {
    #     'fly_from': 'FRA',
    #     'fly_to': ,
    #     'date_from': '01/04/2021',
    #     'date_to': '03/04/2021'
    # }

    # Sending GET request
    response = requests.get(url, headers=headers, params=params)

    # Checking response status
    if response.status_code == 200:
        print("Request successful!")
        # Printing response data
        print(response.json())
    else:
        print("Request failed with status code:", response.status_code)
    
def extract_city(user_message):
    # Regular expression to match patterns like "Weather forecast for [city]"
    pattern = r'[^a-zA-Z0-9\s]'  # Keep alphanumeric characters and whitespace

    # Use the sub() function to replace matched special characters with an empty string
    clean_text = re.sub(pattern, '', user_message)
    for word in clean_text.split(" "):
        city = word.capitalize()
        if city in top_100_countries:
            return city
    return None


def weather(city):
    res = ""

    # Enter your API key here
    api_key = "2adc25233f588c7b0bf1052ddd5c1947"

    # base_url variable to store url
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    # Give city name
    city_name = city

    # complete_url variable to store
    # complete url address
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name

    # get method of requests module
    # return response object
    response = requests.get(complete_url)

    # json method of response object 
    # convert json format data into
    # python format data
    x = response.json()

    # Now x contains list of nested dictionaries
    # Check the value of "cod" key is equal to
    # "404", means city is found otherwise,
    # city is not found
    if x["cod"] != "404":

        # store the value of "main"
        # key in variable y
        y = x["main"]

        # store the value corresponding
        # to the "temp" key of y
        current_temperature = y["temp"]

        # store the value corresponding
        # to the "pressure" key of y
        current_pressure = y["pressure"]

        # store the value corresponding
        # to the "humidity" key of y
        current_humidity = y["humidity"]

        # store the value of "weather"
        # key in variable z
        z = x["weather"]

        # store the value corresponding 
        # to the "description" key at 
        # the 0th index of z
        weather_description = z[0]["description"]

        # print following values
        print(
              " Temperature (in kelvin unit) = " +
                        str(current_temperature) +
            "\n atmospheric pressure (in hPa unit) = " +
                        str(current_pressure) +
            "\n humidity (in percentage) = " +
                        str(current_humidity) +
            "\n description = " +
                        str(weather_description))
        
        res += city_name +  ": Temperature (in kelvin unit) = " + str(current_temperature) + "\n atmospheric pressure (in hPa unit) = " + str(current_pressure) + "\n humidity (in percentage) = " + str(current_humidity) + "\n description = " + str(weather_description)
        return res
    else:
        print(" City Not Found ")

def convert_currency(amount, from_currency, to_currency):
    # Replace 'YOUR_APP_ID' with your actual Open Exchange Rates App ID
    app_id = 'b180edd53e54401baa3d24f3927f348a'
    
    # Construct the API URL
    url = f"https://open.er-api.com/v6/latest/{from_currency}"
    
    try:
        # Make a GET request to the API
        response = requests.get(url, params={'app_id': app_id})
        data = response.json()
        
        # Check if the request was successful
        if response.status_code == 200:
            # Extract the exchange rate from the response data
            exchange_rate = data['rates'][to_currency]
            
            # Convert the amount
            converted_amount = amount * exchange_rate
            return converted_amount
        
        else:
            # Print error message if request was not successful
            print(f"Error: {data['error']['message']}")
            return None
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def get_currency(country_name):
    # Convert country name to title case for case-insensitive matching
    country_name = country_name.title()
    
    if country_name in country_currency:
        return country_currency[country_name]
    else:
        return "US"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.form['user_message']
        
    intents_list = predict_class(user_message)
    response = get_response(intents_list, intents, user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
