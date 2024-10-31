
import os
import re
import random
import json
import pickle
import time
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import telebot
import requests
from telebot import types

# Load the chatbot model and data
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intense.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

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

def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    try:
        bow = bag_of_words(sentence)  # Ensure this matches the model's expected input
        print("Bow: ", bow)
        res = model.predict(np.array([bow]))[0]  # Predict and take the first result
        print("Model output:", res)  # Debug: print raw model output
        ERROR_THRESHOLD = 0.25  # Lower threshold if model's confidence is generally low
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        if not return_list:
            print("No intents exceed the threshold of:", ERROR_THRESHOLD)
        # print("Filtered and sorted results:", return_list)  # Debug: print processed results
        return return_list
    except Exception as e:
        print("An error occurred in predict_class:", e)
        return []  # Optionally, return an empty list or a specific error message



def get_response(intents_list, intents_json, user_message, tag):
    if(tag != user_message):
        tag = intents_list[0]['intent']
    else:
        tag = user_message

    list_of_intents = intents_json['intents']
    result = ""
    print("Handling intent tagged as:", tag)

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            if tag == "get_country_currency":
                country = extract_city(user_message)
                if country:
                    currency = country_currency[country]
                    result += " " + currency
                else:
                    result = "Sorry, I couldn't identify the Country."
            return result
    
    # If no suitable intent response was found
    return "I'm sorry, I didn't understand that. Could you please rephrase or try asking something else?"


def get_emergency_numbers(country_name):
    url = "https://countrywise.p.rapidapi.com/"

    querystring = {"country": country_name}

    headers = {
        "X-RapidAPI-Key": "470f880cc5mshed34ef21ce87df8p1d3a01jsnae30814bce78",
        "X-RapidAPI-Host": "countrywise.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.text)
        return None

def get_country_code(country_name):
    # Remove spaces from country_name and make it lowercase
    country_name = country_name.replace(" ", "").lower()
    print(country_name)
    country_codes = {
        "canada": "ca",
        "unitedstates": "us",
        "australia": "au",
        "unitedkingdom": "uk",
        "germany": "de",
        "france": "fr",
        "japan": "jp",
        # Add more countries as needed
    }

    return country_codes.get(country_name, "")


def extract_city(user_message):
    pattern = r'[^a-zA-Z0-9\s]'
    clean_text = re.sub(pattern, '', user_message)
    for word in clean_text.split(" "):
        city = word.capitalize()
        if city in top_100_countries:
            return city
    return None


def weather(city):
    res = ""
    api_key = "2adc25233f588c7b0bf1052ddd5c1947"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    city_name = city
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]
        current_temperature = y["temp"]
        current_pressure = y["pressure"]
        current_humidity = y["humidity"]
        z = x["weather"]
        weather_description = z[0]["description"]

        res += city_name + ": Temperature (in kelvin unit) = " + str(current_temperature) + "\n atmospheric pressure (in hPa unit) = " + str(
            current_pressure) + "\n humidity (in percentage) = " + str(current_humidity) + "\n description = " + str(weather_description)
        return res
    else:
        print(" City Not Found ")


def convert_currency(amount, from_currency, to_currency):
    app_id = 'b180edd53e54401baa3d24f3927f348a'
    url = f"https://open.er-api.com/v6/latest/{from_currency}"
    try:
        response = requests.get(url, params={'app_id': app_id})
        data = response.json()
        if response.status_code == 200:
            exchange_rate = data['rates'][to_currency]
            converted_amount = amount * exchange_rate
            return converted_amount

        else:
            print(f"Error: {data['error']['message']}")
            return None

    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def get_currency(country_name):
    country_name = country_name.title()
    if country_name in country_currency:
        return country_currency[country_name]
    else:
        return "US"


# Telegram bot token
BOT_TOKEN = "6764597929:AAGxe5Ia_bOh22zIImcQ2oSUr0GFIm2tA4Q"

# Initialize the bot
bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    welcome_message = (
        "Welcome to Smriti Travel Bot! Currently, our services are available for the following countries only: Usa, Canada, France, UK, and India.\n\n"
        "Here are some commands you can use:\n"
        "/Weather - Get weather information.\n"
        "/EmergencyNumbers - Retrieve emergency numbers.\n"
        "/Covid - Check Covid-19 guidelines.\n"
        "/LocalRecommendations - Get local recommendations.\n"
        "/TravelBudget - Estimate the travel budget for your destination.\n"
        "/HotelBooking - Book hotels.\n"
        "/CustomerExperiences - Hear what others have experienced.\n"
        "/LocalTransportation - Information about local transportation options.\n"
        "/booking_flight - Book a flight to your destination.\n"
        "/Currency - Get Currency for Country.\n\n",
    )
    bot.reply_to(message, welcome_message)


# Currency Handler
@bot.message_handler(commands=['Currency'])
def weather_handler(message):
    bot.reply_to(message, "Please enter the Country for Currency:")
    bot.register_next_step_handler(message, curr_ask_city)

# Function to handle city input
def curr_ask_city(message):
    country = message.text.lower()
    user_intent = f"Currency/{country.capitalize()}"
    intents_list = predict_class(user_intent)
    if(intents_list[0]["intent"] != user_intent):
        intents_list[0] = {'intent': user_intent}
    response = get_response(intents_list, intents, user_intent, tag=user_intent)
    bot.reply_to(message, response)


# Weather Handler
@bot.message_handler(commands=['Weather'])
def weather_handler(message):
    bot.reply_to(message, "Please enter the Country for weather information:")
    bot.register_next_step_handler(message, ask_city)

# Function to handle city input
def ask_city(message):
    country = message.text.lower()
    result = weather(country)
    user_intent = "weather_information"
    intents_list = predict_class("Weather forecast for [city]")
    if(intents_list[0]["intent"] != user_intent):
        intents_list[0] = {'intent': user_intent}
    response = get_response(intents_list, intents, user_intent, tag=user_intent)
    response += result
    bot.reply_to(message, response)


@bot.message_handler(commands=['EmergencyNumbers'])
def emergency_numbers_handler(message):
    bot.reply_to(message, "Please enter the country for Emergency Numbers:")
    bot.register_next_step_handler(message, ask_emergency_numbers_country)

# Function to handle country input for emergency numbers
def ask_emergency_numbers_country(message):
    country = message.text.lower()
    user_intent = f"EmergencyNumbers/{country.capitalize()}"
    intents_list = predict_class(user_intent)
    if(intents_list[0]["intent"] != user_intent):
        intents_list[0] = {'intent': user_intent}
    response = get_response(intents_list, intents, user_intent, tag=user_intent)
    bot.reply_to(message, response)


# Covid Handler
@bot.message_handler(commands=['Covid'])
def covid_handler(message):
    bot.reply_to(message, "Please enter the country for Covid Guidelines:")
    bot.register_next_step_handler(message, covid_ask_country)

# Function to handle country input
def covid_ask_country(message):
    country = message.text.lower()
    user_intent = f"Covid/{country.capitalize()}"
    intents_list = predict_class(user_intent)
    if(intents_list[0]["intent"] != user_intent):
        intents_list[0] = {'intent': user_intent}
    response = get_response(intents_list, intents, user_intent, tag=user_intent)
    bot.reply_to(message, response)

# Local Recommendations Handler
@bot.message_handler(commands=['LocalRecommendations'])
def local_recommendations_handler(message):
    bot.reply_to(message, "Please enter the country for Local Recommendations:")
    bot.register_next_step_handler(message, local_ask_country)

# Function to handle country input
def local_ask_country(message):
    country = message.text.lower()
    user_intent = f"LocalRecommendations/{country.capitalize()}"
    intents_list = predict_class(user_intent)
    if(intents_list[0]["intent"] != user_intent):
        intents_list[0] = {'intent': user_intent}
    response = get_response(intents_list, intents, user_intent, tag=user_intent)
    bot.reply_to(message, response)

# Travel Budget Handler
@bot.message_handler(commands=['TravelBudget'])
def send_welcome(message):
    bot.reply_to(message, "Please enter the country for Travel Budget:")
    bot.register_next_step_handler(message, travel_ask_country)


# Function to handle country input
def travel_ask_country(message):
    country = message.text.lower()
    user_intent = f"TravelBudget/{country.capitalize()}"
    intents_list = predict_class(user_intent)
    if(intents_list[0]["intent"] != user_intent):
        intents_list[0] = {'intent': user_intent}
    response = get_response(intents_list, intents, user_intent, tag=user_intent)
    bot.reply_to(message, response)


# Hotel Booking Handler
@bot.message_handler(commands=['HotelBooking'])
def hotel_booking_handler(message):
    bot.reply_to(message, "Please enter the country for Hotel Booking:")
    bot.register_next_step_handler(message, hotel_ask_country)

# Function to handle country input
def hotel_ask_country(message):
    country = message.text.lower()
    user_intent = f"HotelBooking/{country.capitalize()}"
    intents_list = predict_class(user_intent)
    if(intents_list[0]["intent"] != user_intent):
        intents_list[0] = {'intent': user_intent}
    response = get_response(intents_list, intents, user_intent, tag=user_intent)
    bot.reply_to(message, response)

# Customer Experiences Handler
@bot.message_handler(commands=['CustomerExperiences'])
def customer_experiences_handler(message):
    bot.reply_to(message, "Please enter the country for Customer Experiences:")
    bot.register_next_step_handler(message, cus_ask_country)

# Function to handle country input
def cus_ask_country(message):
    country = message.text.lower()
    user_intent = f"CustomerExperiences/{country.capitalize()}"
    intents_list = predict_class(user_intent)
    if(intents_list[0]["intent"] != user_intent):
        intents_list[0] = {'intent': user_intent}
    response = get_response(intents_list, intents, user_intent, tag=user_intent)
    bot.reply_to(message, response)

# Local Transportation Handler
@bot.message_handler(commands=['LocalTransportation'])
def local_transportation_handler(message):
    bot.reply_to(message, "Please enter the country for Local Transportation:")
    bot.register_next_step_handler(message, trans_ask_country)

# Function to handle country input
def trans_ask_country(message):
    country = message.text.lower()
    user_intent = f"LocalTransportation/{country.capitalize()}"
    intents_list = predict_class(user_intent)
    if(intents_list[0]["intent"] != user_intent):
        intents_list[0] = {'intent': user_intent}
    response = get_response(intents_list, intents, user_intent, tag=user_intent)
    bot.reply_to(message, response)


# Dictionary to store user details temporarily
user_details = {}

# Booking Handler
@bot.message_handler(commands=['booking_flight'])
def send_welcome(message):
    bot.reply_to(message, "Please enter your source destination:")
    bot.register_next_step_handler(message, ask_source_destination)
    
# Function to handle source destination input
def ask_source_destination(message):
    source_destination = message.text

    # Store the source destination
    user_details['source_destination'] = source_destination
    
    # Ask the user for final destination
    bot.reply_to(message, "Please enter your final destination:")
    bot.register_next_step_handler(message, ask_final_destination)

# Function to handle final destination input
def ask_final_destination(message):
    final_destination = message.text

    # Store the final destination
    user_details['final_destination'] = final_destination
    
    # Ask the user for check-in date
    bot.reply_to(message, "Please enter your travel date (YYYY-MM-DD):")
    bot.register_next_step_handler(message, ask_checkin_date)

# Function to handle check-in date input
def ask_checkin_date(message):
    checkin_date = message.text
    
    # Store the check-in date
    user_details['checkin_date'] = checkin_date
    
    # Ask the user for check-out date
    bot.reply_to(message, "Please enter your return date (YYYY-MM-DD):")
    bot.register_next_step_handler(message, ask_checkout_date)

# Function to handle check-out date input
def ask_checkout_date(message):
    checkout_date = message.text

    # Store the check-out date
    user_details['checkout_date'] = checkout_date
    
    # Ask the user for the number of guests
    bot.reply_to(message, "Please enter the number of guests:")
    bot.register_next_step_handler(message, confirm_booking_details)

# Function to confirm booking details
def confirm_booking_details(message):
    num_guests = message.text
    
    # Store the number of guests
    user_details['num_guests'] = num_guests
    
    # Retrieve all the details
    source_destination = user_details['source_destination']
    final_destination = user_details['final_destination']
    checkin_date = user_details['checkin_date']
    checkout_date = user_details['checkout_date']
    num_guests = user_details['num_guests']
    
    # Generate response based on user details
    response = f"You have entered the following details:\n\nSource Destination: {source_destination}\nFinal Destination: {final_destination}\nCheck-in Date: {checkin_date}\nCheck-out Date: {checkout_date}\nNumber of Guests: {num_guests}\n\nThank you for providing the details. We will process your booking request."
    
    # Send response to the user
    bot.reply_to(message, response)
    bot.register_next_step_handler(message, get_flight_search_results(message=message, fly_from=source_destination, fly_to=final_destination, date_to=checkin_date, date_from=checkout_date))



def get_flight_search_results(message, fly_from, date_from, fly_to, date_to):

    fly_from = countryCode(fly_from)
    fly_to = countryCode(fly_to)

    print(fly_to)

    url = "https://tripadvisor16.p.rapidapi.com/api/v1/flights/searchFlights"

    querystring = {"sourceAirportCode": fly_from,"destinationAirportCode":fly_to,"date":date_from,"itineraryType":"ONE_WAY","sortOrder":"PRICE","numAdults":"1","numSeniors":"0","classOfService":"ECONOMY","pageNumber":"1","currencyCode":"USD", "returnData": date_to}

    headers = {
        "X-RapidAPI-Key": "5e3aa98385mshf4ca307887996c2p1c5e5fjsn9073021c1a48",
        "X-RapidAPI-Host": "tripadvisor16.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()["data"]
    res= ""
    count = 0
    for flight in data['flights']:
        if count >= 3:
            break  # Stop the loop if three flights have been processed
        for segment in flight['segments']:
            for leg in segment['legs']:
                res += f"Origin: {leg['originStationCode']} -> Destination: {leg['destinationStationCode']}\n"
                res += f"Departure: {leg['departureDateTime']} -> Arrival: {leg['arrivalDateTime']}\n"
            res += "--- Segment Details End ---\n"
        for purchaseLink in flight['purchaseLinks']:
            res += f"Total Price Per Passenger: {purchaseLink['totalPricePerPassenger']}\n"
            res += f"Booking Link: {purchaseLink['url']}\n"
            res += "--- Purchase Link Details End ---\n"
        count += 1  # Increment count for each flight processed
    
    print(res)
    if(res):
        bot.reply_to(message, res)

def countryCode(country):
    url = "https://tripadvisor16.p.rapidapi.com/api/v1/flights/searchAirport"

    querystring = {"query":country.lower()}

    headers = {
        "X-RapidAPI-Key": "5e3aa98385mshf4ca307887996c2p1c5e5fjsn9073021c1a48",
        "X-RapidAPI-Host": "tripadvisor16.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    return data["data"][0]["airportCode"]

# Handler for all other messages
@bot.message_handler(func=lambda msg: True)
def handle_message(message):
    user_message = message.text
    print(user_message)
    intents_list = predict_class(user_message)

    print("Intent List: ", intents_list)
    response = get_response(intents_list, intents, user_message, tag=None)
    bot.reply_to(message, response)



# Run the bot
bot.polling()

