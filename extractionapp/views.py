from django.shortcuts import render
from django.core.cache import cache
import requests
from .config import API_CONFIG, WHISPER_API_CONFIG,GROK_API_CONFIG,RAZORPAY_API_KEY, RAZORPAY_API_SECRET,DATAMUSE_API_URL, DATAMUSE_RELATED_URL,KEYWORD_API_BASE_URL,USERINFO_ENDPOINT,STABLEDIFFUSION_API_CONFIG
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import urlparse, parse_qs
import os
import subprocess
import whisper
import spacy
import openai
import random
import json

import logging

logger = logging.getLogger(__name__)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
from sklearn.feature_extraction.text import CountVectorizer

from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import transformers

import torch


from .models import Keyword,videoSEODB,urlSEODB,YouTubeUser,Subscription_Data,Subscription,ContactForm,PaymentDetails,AffiliateUser,Payment,PaymentPlan
from django.conf import settings

from youtube_transcript_api import YouTubeTranscriptApi
import re

from django.conf import settings
from django.shortcuts import redirect,get_object_or_404
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

import razorpay

from django.db import transaction
from pytrends.request import TrendReq

import time
# After each request to pytrends, wait for a few seconds
time.sleep(5)

import asyncio
from .scrapy_run import YouTubeSpiderRunner

import sys
# Add the Scrapy project directory to sys.path so Scrapy can be accessed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fozato_scrapy_project'))


scraped_data = []  # To store scraped results

from django.core.management import call_command
from io import StringIO

from urllib.parse import urlencode
from parsel import Selector
import urllib.parse

from youtubesearchpython import VideosSearch

import base64
from io import BytesIO
from PIL import Image
import uuid

# free trial
from django.utils import timezone
from datetime import timedelta

import warnings
from django.core.cache.backends.base import CacheKeyWarning
# to convert payment amount string to decimal
from decimal import Decimal

# Import your Scrapy spider
# from fozato_scrapy_project.spiders.youtube_tags import YouTubeTagsSpider

# Now import the spider
# from fozato_scrapy_project.spiders.youtube_tags import YouTubeTagsSpider


# Initialize Razorpay client
# razorpay_client = razorpay.Client(auth=(RAZORPAY_API_KEY, RAZORPAY_API_SECRET))
from razorpay import Client, Utility

from django.http import HttpResponse
from razorpay.errors import SignatureVerificationError

# YouTube OAuth 2.0 Scopes
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

# Path to the Google OAuth credentials JSON file (save this in the root folder)
CREDENTIALS_PATH = os.path.join(settings.BASE_DIR, "credentials.json")

# Start the OAuth flow
flow = Flow.from_client_secrets_file(
    CREDENTIALS_PATH,
    scopes=SCOPES,
    # redirect_uri="https://localhost:8000/auth/callback"
    redirect_uri="https://ada3-103-161-55-72.ngrok-free.app/auth/callback"

)
# Generate the authorization URL
auth_url, _ = flow.authorization_url(prompt='consent')

# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
PATTERN = r'%{WORD:keyword}'

#cache = {}
#last_request_time = 0




def indexpage(request):
    return render(request, 'index.html')
def index_page(request):
    return render(request, 'index_page.html') 
def seo_options(request):
#     proxies = [
#     'http://133.18.234.13:80',
#     'http://160.86.242.23:8080',
#     'http://212.107.28.120:80',
#     'http://211.128.96.206:80',
#     'http://68.185.57.66:80',
#     'http://50.231.104.58:80'
# ]

#     for proxy in proxies:
#         try:
#             response = requests.get('http://httpbin.org/ip', proxies={'http': proxy, 'https': proxy}, timeout=5)
#             print(f"Proxy {proxy} works! Response: {response.json()}")
#         except Exception as e:
#             print(f"Proxy {proxy} failed: {e}")
    return render(request,'seooptions.html')
# def home(request):
#     return render(request, 'home.html')  
def goal_selection(request):
    goal_options = {
        'Boost video views': 'Boost video views',
        'Increase sales': 'Increase sales',
        'Increase ad revenue': 'Increase ad revenue',
        'Increase subscribers': 'Increase subscribers',
        'Increase watch time': 'Increase watch time',
        'Improve video SEO': 'Improve video SEO'
    }
    return render(request,'goalselection.html', {'goal_options': goal_options})

def discovery_selection(request):
    # Dictionary of discovery options
    discovery_options = {
        'Ad': 'Ad',
        'Google': 'Google',
        'Facebook': 'Facebook',
        'YouTube': 'YouTube',
        'Linkedin': 'LinkedIn',
        'Twitter': 'Twitter',
        'Referral': 'Referral',
        'Other': 'Other'
    }
    
    return render(request, 'discoveryselection.html', {'discovery_options': discovery_options})
def paymentselection(request):
    return render(request,'paymentselection.html')

def samplepage(request):
    return render(request,'samplepage.html')
def checkout(request):
    return render(request,'checkout.html')
def affiliate_dashboard(request):
    return render(request,'dashboard.html')   
def referal_analytics(request):
    return render(request,'aff_dashboard_ref_analytics.html')
def mobilenumber(request):
    return render(request,'mobilenumber.html')     
def user_dashboard(request):
    return render(request,'user_dashboard1.html')
def user_dahboard_settings(request):
    return render(request,'user_dashboard_settings.html')

from django.http import JsonResponse
from io import StringIO
from urllib.parse import urlencode
from django.core.management import call_command

import hashlib

def generate_referral_code(email):
    """Generate a unique referral code based on email."""
    return hashlib.md5(email.encode()).hexdigest()[:8]  # Shorten to 8 characters

def generate_referral(request):
    print("HTTP method received:", request.method)  # Debugging
    if request.method == 'POST':
        email = request.POST.get('email')
        print("Email received:", email)  # Debugging

        if not email:
            return JsonResponse({'error': 'Email is required'}, status=400)

        # Check if user already has a referral code
        affiliate_user, created = AffiliateUser.objects.get_or_create(email=email)
        if created:
            affiliate_user.referral_code = generate_referral_code(email)
            affiliate_user.save()
         
        # Construct the referral link to point to the home page
        referral_link = f"https://127.0.0.1:8000/home?referral_code={affiliate_user.referral_code}"


        print("Referral code:", affiliate_user.referral_code)  # Debugging
        return JsonResponse({'referral_code': referral_link})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)



class FozatoDataExtractor:
    def __init__(self, text):
        self.text = text
        self.keywords = []

    def generate_keywords_from_google_trends(self):
        """
        Fetch related keywords using Google Trends and return them as JSON.
        """
        keywords_input = self.text
        

        try:
            
            # Split the keywords by commas and clean extra whitespace
            main_keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
            
            # Check if keywords are provided
            if not main_keywords:
                return JsonResponse({'status': 'error', 'message': 'No keywords provided.'}, status=400)
            
            # Initialize pytrends
            pytrend = TrendReq(hl='en-US', tz=360)

            # Dictionary to store related keywords
            related_keywords = {}

            for keyword in main_keywords:
                # Get suggestions for the keyword
                suggestions = pytrend.suggestions(keyword=keyword)
                related_keywords[keyword] = [s["title"] for s in suggestions]  # Extract titles only

            return JsonResponse({'status': 'success', 'data': related_keywords}, status=200)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    

    def extraction_from_text_api(self):
        """
        Generate suggestions and thumbnails based on the input text.
        """
        text = self.text
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)

        try:
            # Generate suggestions based on the text entered
            suggestions = YouTubeVideo.generate_suggestions_for_prefixes(text)
            keywords = [suggestion[0] for key in suggestions for suggestion in suggestions[key]]
            print("api keywords:",keywords)

            # Call the Stable Diffusion API to generate thumbnails
            # thumbnails = YouTubeVideo.generate_thumbnails_from_stable_diffusion(keywords[0])

            return JsonResponse({"keywords": keywords})

        except Exception as e:
            return JsonResponse({"error": str(e)})

    def extraction_from_text_trends(self):
        """
        Execute Scrapy spider for trend extraction.
        """
        text = self.text
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)

        try:
            # Capture Scrapy output using StringIO
            out = StringIO()
            call_command('run_trends', stdout=out)
            spider_output = out.getvalue()

            return JsonResponse({'spider_output': spider_output.strip()})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    def extraction_from_text_scrapy(self):
        """
        Extract YouTube video details and keywords using Scrapy.
        """
        text = self.text
        if not text:
            return JsonResponse({"error": "No text provided"}, status=400)

        try:
            # Generate YouTube search URL
            query_params = {"search_query": text}
            search_url = f"https://www.youtube.com/results?{urlencode(query_params)}"

            # Fetch YouTube video URLs via helper function
            video_urls = YouTubeVideo.get_youtube_video_urls(text)

            if not video_urls:
                return JsonResponse({"error": "No video URLs found for the given search text."}, status=404)

            # Extract paths and filter valid video links
            video_paths = [url.split("youtube.com")[-1] for url in video_urls]
            full_video_urls = [
                f"https://www.youtube.com{path}" for path in video_paths if "/watch?v=" in path
            ]

            if not full_video_urls:
                return JsonResponse({"error": "No valid video paths found."}, status=404)

            # Create an instance of YouTubeSpiderRunner for a single URL
            spider_runner = YouTubeSpiderRunner()
            result_using_scrapy = spider_runner.run_spider_for_multiple_urls(full_video_urls)

            return JsonResponse(
                {"message": "Scrapy spider ran successfully", "data": result_using_scrapy},
                status=200
            )

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    def search_keywords_from_google_trends(self, keywords_input):
        """
        Fetch related keywords using Google Trends and return them as JSON.
        """
        try:
            # Split the keywords by commas and clean extra whitespace
            main_keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
            
            # Check if keywords are provided
            if not main_keywords:
                return JsonResponse({'status': 'error', 'message': 'No valid keywords provided.'}, status=400)
            
            # Log the keywords for debugging
            print("Keywords to fetch trends for:", main_keywords)
            
            # Initialize pytrends
            pytrend = TrendReq(hl='en-US', tz=360)

            # Dictionary to store related keywords
            related_keywords = {}

            for keyword in main_keywords:
                try:
                    # Get suggestions for the keyword
                    suggestions = pytrend.suggestions(keyword=keyword)
                    
                    # If no suggestions are found, log it
                    if not suggestions:
                        print(f"No suggestions found for keyword: {keyword}")
                    
                    related_keywords[keyword] = [s["title"] for s in suggestions]  # Extract titles only

                except Exception as e:
                    print(f"Error fetching suggestions for {keyword}: {str(e)}")

            return JsonResponse({'status': 'success', 'data': related_keywords}, status=200)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def extraction_from_text(request):
    if request.method == "POST":
        text = request.POST.get("text")
        
        if not text:
            return JsonResponse({"error": "No text provided"}, status=400)

        # Create an instance of the extractor with the provided text
        kw_extractor = FozatoDataExtractor(text)
        
        # Call the extraction functions and capture their results
        result_scrapy = kw_extractor.extraction_from_text_scrapy()
        result_trends = kw_extractor.generate_keywords_from_google_trends()
        result_api = kw_extractor.extraction_from_text_api()

        # Extract the content of JsonResponse objects as JSON-compatible data
        result_scrapy_data = json.loads(result_scrapy.content)
        result_trends_data = json.loads(result_trends.content)
        result_api_data = json.loads(result_api.content)

        print("Scrapy result:", result_scrapy_data)
        print("result_trends:", result_trends_data)
        print("result_api:", result_api_data)

        # Check if the trends extraction was successful based on the returned dictionary
        if "error" in result_trends_data:
            return JsonResponse({"error": "Trends extraction failed", "details": result_trends_data["error"]}, status=500)

        # Concatenate and deduplicate results
        all_keywords = []

        # Collect keywords from Scrapy data if present
        if "data" in result_scrapy_data and isinstance(result_scrapy_data["data"], list):
            all_keywords.extend(result_scrapy_data["data"])

        # Collect keywords from trends data if present
        if "data" in result_trends_data and isinstance(result_trends_data["data"], dict):
            for key, keywords in result_trends_data["data"].items():
                all_keywords.extend(keywords)

        # Collect keywords from API data if present
        if "keywords" in result_api_data and isinstance(result_api_data["keywords"], list):
            all_keywords.extend(result_api_data["keywords"])

            print("All Keywords before deduplication:", all_keywords)
    

        # Remove duplicates while preserving order
        unique_keywords = list(dict.fromkeys(
            keyword if isinstance(keyword, str) else str(keyword) for keyword in all_keywords
        ))

        # Convert list into text
        unique_keywords_text = ", ".join(unique_keywords)
        # Assuming FozatoDataExtractor is a class, create an instance first
        data_extractor = FozatoDataExtractor(unique_keywords_text)
        print("unique keywords:",unique_keywords_text)

        # Call the method with the correct argument
        trend_kw = data_extractor.search_keywords_from_google_trends(unique_keywords_text)
        print("trend_kw:",trend_kw)
        result_from_trends_kw = json.loads(trend_kw.content)
        print("result kw:",result_from_trends_kw)

        # Check if the response has 'data' (or another key you expect), and if its length exceeds 500 characters
        tag_list = []
        if "data" in result_from_trends_kw and isinstance(result_from_trends_kw["data"], str):
            trends_data = result_from_trends_kw["data"]
            
            # If the trends data exceeds 500 characters, split it
            if len(trends_data) > 500:
                print("LENGTH OF TRENDS_DATA:",len(trends_data))
                main_data = trends_data[:500]  # First 500 characters
                tag_list.append(trends_data[500:])  # Remaining characters after 500
            else:
                main_data = trends_data  # Use all if it's less than or equal to 500 characters

            # Update the result with the main data and the tag list
            result_from_trends_kw["data"] = main_data
            result_from_trends_kw["tag_list"] = tag_list

            print("TAGS:",result_from_trends_kw["tag_list"])

                
        # Return the combined and deduplicated results
        return JsonResponse({
            "Main data":result_from_trends_kw["data"],
            "TAGLIST": tag_list,
           "result_from_trends_kw":result_from_trends_kw,
            "combined_keywords": unique_keywords,
            "scrapy_result": result_scrapy_data,
            "trends_result": result_trends_data,
            "api_result": result_api_data,
        }, status=200)
    
@csrf_exempt
def extraction_from_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video_file')
        if not video_file:
            return JsonResponse({'error':'Please choose a file'}, status=400)
        
        else:
            # If it exist, process the video file and generate new data
            try:
                audio_file_path = YouTubeVideo.process_video_file(video_file)
                logger.info("Audio file path: %s", audio_file_path)

                logger.info("About to transcribe audio from: %s", audio_file_path)
                transcription = YouTubeVideo.transcribe_audio(audio_file_path)
                print("Transcription:",transcription)
                logger.info("Transcription completed.",transcription)

                # Limit the transcription to the first 2000 characters,bcz google suggest api has limitation.it allow only 2024 chars
                
                # URL encode the text
                # transcription = urllib.parse.quote(transcription_large_text[:50])  
                

                # Generate keywords from transcription
                keywords_with_volume = YouTubeVideo.fetch_keywords_from_apis(transcription)
                
                keywords = [kw[0] for kw in keywords_with_volume]
                print("Keywords are:...",keywords)
                
               
                # Create an instance of the extractor with the provided text
                kw_extractor = FozatoDataExtractor(transcription)
                
                # Call the extraction functions and capture their results
                result_scrapy = kw_extractor.extraction_from_text_scrapy()
                result_trends = kw_extractor.generate_keywords_from_google_trends()
                # result_api = kw_extractor.extraction_from_text_api()
                # print("result_scrapy:",result_scrapy)



                # # Extract the content of JsonResponse objects as JSON-compatible data
                result_scrapy_data = json.loads(result_scrapy.content)
                result_trends_data = json.loads(result_trends.content)
                # result_api_data = json.loads(result_api.content)

                print("Scrapy result:", result_scrapy_data)
                print("result_trends:", result_trends_data)
                # print("result_api:", result_api_data)

                # Check if the trends extraction was successful based on the returned dictionary
                if "error" in result_trends_data:
                    return JsonResponse({"error": "Trends extraction failed", "details": result_trends_data["error"]}, status=500)

                # Concatenate and deduplicate results
                all_keywords = []

                # Collect keywords from Scrapy data if present
                if "data" in result_scrapy_data and isinstance(result_scrapy_data["data"], list):
                    all_keywords.extend(result_scrapy_data["data"])

                # Collect keywords from trends data if present
                if "data" in result_trends_data and isinstance(result_trends_data["data"], dict):
                    for key, keywords in result_trends_data["data"].items():
                        all_keywords.extend(keywords)

                # Collect keywords from API data if present
                # if "keywords" in result_api_data and isinstance(result_api_data["keywords"], list):
                #     all_keywords.extend(result_api_data["keywords"])

                    print("All Keywords before deduplication:", all_keywords)
               
                # Remove duplicates while preserving order
                # unique_keywords = list(dict.fromkeys(
                #     keyword if isinstance(keyword, str) else str(keyword) for keyword in all_keywords
                # ))

                # Convert list into text
                # unique_keywords_text = ", ".join(unique_keywords)
                # Assuming FozatoDataExtractor is a class, create an instance first
                # data_extractor = FozatoDataExtractor(unique_keywords_text)
                # print("unique keywords:",unique_keywords_text)

                # Call the method with the correct argument
                # trend_kw = data_extractor.search_keywords_from_google_trends(unique_keywords_text)
                # print("trend_kw:",trend_kw)
                # result_from_trends_kw = json.loads(trend_kw.content)
                # print("result kw:",result_from_trends_kw)

                # Check if the response has 'data' (or another key you expect), and if its length exceeds 500 characters
                # tag_list = []
                # if "data" in result_from_trends_kw and isinstance(result_from_trends_kw["data"], str):
                #     trends_data = result_from_trends_kw["data"]
                    
                #     # If the trends data exceeds 500 characters, split it
                #     if len(trends_data) > 500:
                #         print("LENGTH OF TRENDS_DATA:",len(trends_data))
                #         main_data = trends_data[:500]  # First 500 characters
                #         tag_list.append(trends_data[500:])  # Remaining characters after 500
                #     else:
                #         main_data = trends_data  # Use all if it's less than or equal to 500 characters

                #     # Update the result with the main data and the tag list
                #     result_from_trends_kw["data"] = main_data
                #     result_from_trends_kw["tag_list"] = tag_list

                #     print("TAGS:",result_from_trends_kw["tag_list"])

                        
                # Return the combined and deduplicated results
                return JsonResponse({
                  
                    "scrapy_result": result_scrapy_data,
                    "trends_result": result_trends_data,
                    "api_result": keywords,
                }, status=200)    
            except Exception as e:
                return JsonResponse({'error': 'An error occurred while processing the video.', 'details': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

class YouTubeVideo:
    def __init__(self, url):
        self.url = url
        self.video_id = self.extract_video_id(url)
        self.title = ""
        self.description = ""
        self.tags = []
        self.keywords = []

    def extract_video_id(self, url):
        parsed_url = urlparse(url)
        if 'youtube.com' in parsed_url.netloc:
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif 'youtu.be' in parsed_url.netloc:
            return parsed_url.path.strip('/')
        return None

    def fetch_video_details(self):
        if not self.video_id:
            return None

        cached_result = cache.get(self.video_id)
        if cached_result:
            self.keywords = cached_result
            self.generate_seo_elements()
            return self.keywords

        response = requests.get(
            f'https://www.googleapis.com/youtube/v3/videos?id={self.video_id}&key={API_CONFIG["api_key"]}&part=snippet'
        )
        if response.status_code != 200:
            return JsonResponse({'error': response.json().get('error', 'Unknown error')}, status=response.status_code)
        if response.status_code == 200:
            data = response.json()
            if data['items']:
                snippet = data['items'][0]['snippet']
                self.keywords = snippet.get('tags', [])
                self.generate_seo_elements()
                cache.set(self.video_id, self.keywords, timeout=API_CONFIG['cache_timeout'])
                return self.keywords
        return []

    def generate_seo_elements(self):
        if self.keywords:
            self.title = f"Learn about {', '.join(self.keywords[:3])} - Your Ultimate Guide"
            self.description = f"This video covers {', '.join(self.keywords)}. Watch to learn more about {', '.join(self.keywords[:3])}."
            self.tags = self.keywords[:5]
            return self.title,self.description,self.tags

    @staticmethod

    def process_video_file(video_file):
        upload_dir = 'media/uploads/'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, video_file.name)

        # Ensure the file is not empty before saving
        if video_file.size > 0:
            with open(file_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)
        else:
            raise ValueError("Uploaded file is empty.")

        return file_path

    @staticmethod
    def transcribe_audio(video_file_path):
        # Define audio file path for transcription
        audio_file_path = video_file_path.replace('.mp4', '.wav')
        
        # Extract audio using ffmpeg
        try:
            print(f"Extracting audio from: {video_file_path} using ffmpeg")
            command = ['ffmpeg', '-y', '-i', video_file_path, audio_file_path]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"FFMPEG error: {e}")
        
        # Load Whisper model and transcribe
        try:
            model = whisper.load_model("base")
            # Convert warnings into exceptions
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=CacheKeyWarning)
                result = model.transcribe(audio_file_path)
                print("RESULT IS:", result)
            
            # Check if the detected language is English
            language = result.get('language')
            print("language:", language)  # This will output: 'ml'
            if language != 'en':
                print("error in transcription")
                raise ValueError("Transcription failed due to non-English language. Only English videos can be transcribed.")  # Specific error message
            # Ensure 'text' exists in result
            if 'text' not in result:
                print("no text in result")
                raise ValueError("Transcription failed: 'text' field is missing from result.")
            else:
                print("text in result",result['text'])  
                return result['text']  
            
            
        except CacheKeyWarning as w:
            # Handle cache key warnings explicitly
            raise ValueError("Transcription failed due to invalid characters in cache key.")
        except Exception as e:
            raise ValueError(f"Whisper model error: {e}")
        
        # Clean up by removing audio file after transcription
        os.remove(audio_file_path)
        
        
    

    @staticmethod
    def transcribe_audio_old1(video_file_path):
        audio_file_path = video_file_path.replace('.mp4', '.wav')
        command = ['ffmpeg','-y', '-i', video_file_path, audio_file_path]
        subprocess.run(command, check=True)

        model = whisper.load_model("base")
        result = model.transcribe(audio_file_path)
        return result['text']

    @staticmethod
    def extract_seo_keywords_usingpipeline(text):
        # Load the keyword extraction pipeline
        keyword_extractor = pipeline("feature-extraction", model="distilbert-base-uncased")

    # Extract features
        features = keyword_extractor(text)
    # This example just returns the first token's embeddings as keywords
        keywords = [word for word in features[0][0] if word != 0]
        return keywords


    @staticmethod
    def extract_seo_keywords_secondlast(text):
        try:
            print("TEXT............:",text)
            # Load the model and tokenizer
            model_name = "distilgpt2"
            print(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Tokenizer loaded successfully.")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            print("Model loaded successfully.")  # Confirm successful loading
            print("MODEL......:", model)

            # Prepare input for the model
            prompt = (
                "Analyze the following text and extract keywords along with their estimated search volume. "
                "Format the response as a list of tuples, where each tuple contains a keyword and its search volume. "
                "Here is the text:\n\n"
                f"{text}\n\n"
                "Keywords and their search volume:"
            )
            inputs = tokenizer.encode(prompt, return_tensors='pt',return_attention_mask=True)
            print("INPUTS....:", inputs)

            # Generate keywords
            outputs = model.generate(inputs, max_length=1000)
            print("OUTPUTS.......:", outputs)

            # Decode the generated text
            keywords = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Keywords:", keywords)

            # Optionally, you can further process the keywords
            # For example, split by commas and clean up whitespace
            keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
            return keywords_list
        except Exception as e:
            print("Error:", e)
            return []


    @staticmethod
    def extract_seo_keywords(transcription):
        # Use a correct or available LLaMA model
        model_name = "facebook/llama-2-7b-chat"  # Replace with the correct model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create the prompt for extracting SEO keywords
        prompt = f"Extract the main SEO keywords from the following transcription:\n\n{transcription}\n\nKeywords:"

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate the response from the model
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)

        # Decode the generated output
        keywords_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-process the response to extract keywords
        keywords = []
        if 'Keywords:' in keywords_text:
            # Extracting the portion after 'Keywords:'
            keywords_section = keywords_text.split('Keywords:')[1]
            # Assuming the keywords are comma-separated (this can be adjusted based on model output)
            keywords = [keyword.strip() for keyword in keywords_section.split(',') if keyword.strip()]
        
        return keywords
    
    @staticmethod
    def extract_seo_keywords_nvidialama(transcription):
        # Load the LLaMA model and tokenizer
        model_name = "nvidia/llama-3-1-405b-instruct"  # Replace with the specific model you are using
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        prompt = f"Extract the main SEO keywords from the following transcription:\n\n{transcription}\n\nKeywords:"

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate the response from the model
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)

        # Decode the generated output
        keywords_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract keywords from the response
        # Assuming keywords are separated by commas
        keywords = [keyword.strip() for keyword in keywords_text.split(',') if keyword.strip()]

        return keywords

    @staticmethod
    def extract_seo_keywords_olddd(transcription):
        return YouTubeVideo.analyze_keywords_with_openai(transcription)
    
    @staticmethod
    def extract_seo_keywords_old(transcription):
        # Tokenize the transcription
        tokens = word_tokenize(transcription.lower())
        
        # Remove punctuation and stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [
            word for word in tokens if word not in stop_words and word not in string.punctuation
        ]
        
        # Count word frequencies
        word_counts = Counter(filtered_tokens)
        
        # Get the most common keywords
        common_keywords = word_counts.most_common(10)
        keywords = [word for word, count in common_keywords]
        
        return keywords

    @staticmethod
    def extract_seo_keywords_oldd(transcription):
        """Extract keywords from the transcription."""
    
        # Tokenize the transcription
        tokens = word_tokenize(transcription.lower())
        
        # Remove punctuation and stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [
            word for word in tokens if word not in stop_words and word not in string.punctuation
        ]
        
        # Create n-grams
        n = 2  # Change to 3 for tri-grams
        ngrams = [' '.join(filtered_tokens[i:i+n]) for i in range(len(filtered_tokens)-n+1)]
        
        # Count word frequencies
        word_counts = Counter(filtered_tokens)
        ngram_counts = Counter(ngrams)
        
        # Combine and prioritize relevant keywords
        all_keywords = word_counts.most_common(15) + ngram_counts.most_common(15)
        unique_keywords = list(set([kw[0] for kw in all_keywords if len(kw[0]) > 3]))
        
        return unique_keywords[:15]

        # Get the most common keywords
        #common_keywords = word_counts.most_common(15)  # Adjust number for more or fewer keywords
        #keywords = [word for word, count in common_keywords]
    
          # Prioritize more relevant keywords (you can customize this list based on your content)
        #base_keywords = [kw for kw in keywords if len(kw) > 3]  # Filter out short words
        #return base_keywords[:15]  # Return top 10 relevant keywords
        #related_keywords = []
        #for keyword in base_keywords:
            #for token in tokens:
            #    if keyword in token:
             #       related_keywords.append(f"{keyword} {token}")
        #return list(set(related_keywords))  # Remove duplicates  
    
    @staticmethod
    def get_related_keywords(keyword):
        keyword = requests.GET.get('keyword', '')
        if not keyword:
            return JsonResponse({'error': 'Keyword parameter is required'}, status=400)

        # Format the API URL with the provided keyword
        api_url = DATAMUSE_API_URL.format(keyword=keyword)
        response = requests.get(api_url)
        if response.status_code == 200:
            return [result['word'] for result in response.json()]
        return [] 
    @staticmethod
    def get_ranking_keywords(keyword):
        """Get ranking keywords using the Datamuse API."""
        # Use the 'words' endpoint for more relevant keywords
        api_url = DATAMUSE_RELATED_URL.format(keyword=keyword)
        response = requests.get(api_url)
        if response.status_code == 200:
            return [result['word'] for result in response.json()]
        return []

    
    @staticmethod
    def generate_title(keywords):
        if keywords:
            title = f"Highlights {', '.join(keywords[:3])} - Your Ultimate Guide"
            return title

    @staticmethod
    def generate_description(keywords):
        if keywords:    
            description = f"This video discuss about {', '.join(keywords)}. Watch to learn more about {', '.join(keywords[:3])}."
            return description

    @staticmethod
    def generate_tags(keywords):
        if keywords:        
            tags = keywords[:5]
            return tags
           
    


    @staticmethod
    def analyze_keywords_with_openai(transcription):
        openai.api_key = WHISPER_API_CONFIG["api_key"]  # Your OpenAI API key

        
        prompt = f"Extract relevant SEO keywords from the following transcription:\n\n{transcription}\n\nKeywords:"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4 if you have access
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )

        keywords_text = response['choices'][0]['message']['content']
        keywords = [keyword.strip() for keyword in keywords_text.split(',') if keyword.strip()]
        print("haaaai")
        print("Extracted keywords:", keywords)
        return keywords
    
    @staticmethod
    def extract_keywords_with_grok(transcription):
        """Extract keywords using Grok API."""
        api_url = GROK_API_CONFIG['url']
        api_key = GROK_API_CONFIG['api_key']

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'text': transcription,
            'language': 'en',
            'num_keywords': 10
        }

        try:
            response = requests.post(api_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
        
            if response.headers.get('Content-Type') == 'application/json':
                return response.json()  # Return the JSON response
            else:
                print("Received non-JSON response:", response.text)
                return None
            
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    @staticmethod

    
    def extract_keywords_openai(transcription):
        """Extracts relevant keywords from the transcription using OpenAI API."""
        openai.api_key = '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'
    
        # Prepare the prompt for the API
        prompt = f"Extract relevant SEO keywords from the following transcription:\n\n{transcription}\n\nKeywords:"
    
        try:
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if you have access
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50  # Adjust based on how many keywords you want
            )

            # Extract the text response
            keywords_text = response['choices'][0]['message']['content']
            keywords = [keyword.strip() for keyword in keywords_text.split(',') if keyword.strip()]
        
            return keywords
        except openai.error.OpenAIError as e:
            print(f"An error occurred: {e}")
            return []
        
    #@staticmethod
    #def get_related_keywords_word2vec(keyword):
       # try:
       #     similar_words = model.most_similar(keyword, topn=10)  # Get top 10 similar words
       #     return [word for word, _ in similar_words]
        #except KeyError:
         #   print(f"Keyword '{keyword}' not found in the model.")
         #   return []
   
    


    @staticmethod
    def generate_title_from_keywords_old(keywords):
        return f"Transcription Highlights: {', '.join(keywords[:3])}"

    @staticmethod
    def generate_description_from_keywords(keywords):
        description = f"This video discusses: {', '.join(keywords)}. Dive in for details."
        return description

    @staticmethod
    def filter_relevant_keywords(key_words, min_length=3):
        """Filter keywords to get only relevant ones."""
        key_words = [word.lower() for word in key_words]  # Normalize to lowercase
        relevant_keywords = [
            word for word in key_words 
            if len(word) >= min_length 
        ]
        print("relevant keywords",relevant_keywords)
        print(list(set(relevant_keywords)))
        return (relevant_keywords)   
    @staticmethod
    def generate_multiple_sentence_titles(key_words, max_titles=9):
        """Generate multiple sentence-based titles from keywords."""
        if not key_words:
            return ["No Title Available"]

        # Define a list of templates for generating sentences
        templates = [
            "Explore the world of {}.",
            "Discover amazing techniques for {}.",
            "A comprehensive guide to {} and more.",
            "Unlock the secrets of {}.",
            "Everything you need to know about {}.",
            "Mastering {}: Tips and Tricks.",
            "Your ultimate resource for {}.",
            "The future of {} is here.",
            "How to excel in {}."
        ]

        titles = set()  # Use a set to avoid duplicates

        # Generate titles based on the templates
        while len(titles) < max_titles:
            # Randomly select a template
            selected_template = random.choice(templates)
        
            # Use a combination of keywords to fill the template
            keyword_string = ', '.join(key_words[:10])  # Use first few keywords
            title = selected_template.format(keyword_string)
        
            titles.add(title)

        return list(titles) 
    
    @staticmethod
    def fetch_keywords_from_apis_old(text):
        # Placeholder for keyword extraction logic
        # Use APIs like Google Ads, Ubersuggest, YouTube Data API, etc.
        
        keywords = []
        # Construct the API URL
        api_url = KEYWORD_API_BASE_URL.format(text=text)

        # Example API call to hypothetical keyword API
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('keywords', []):
                keyword = Keyword(keyword=item['keyword'], search_volume=item['search_volume'], source="Example API")
                keyword.save()
                keywords.append((item['keyword'], item['search_volume']))
        
        return keywords    
    @staticmethod
    def fetch_keywords_from_apis(text):
        # Placeholder for keyword extraction logic
        # Use APIs like Google Ads, Ubersuggest, YouTube Data API, etc.

        #text=text
        cached_result = cache.get(text)
        if cached_result:
            keywords = cached_result
            return keywords
        keywords = []

        # YouTube Data API Key
        api_key = API_CONFIG['api_key']  # Make sure to set this in your config.py

        # Search YouTube for relevant keywords
        youtube_url = API_CONFIG['url']
        params = {
            'part': 'snippet',
            'q': text,
            'maxResults': 10,  # Adjust the number of results
            'key': api_key,
        }

        response = requests.get(youtube_url, params=params)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('items', []):
                keyword = item['snippet']['title']
                
                # Set a default value if search_volume cannot be retrieved
                search_volume = 0  # or some other default value

                keywords.append((keyword, search_volume))
                Keyword.objects.create(keyword=keyword, search_volume=search_volume, source="YouTube API")
           
                cache.set(text, keywords, timeout=API_CONFIG['cache_timeout'])

        return keywords
    
    @staticmethod
    def extraction_from_text_trends(text):
   
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        else:
            try:
                 # Create a StringIO object to capture Scrapy output
                out = StringIO()
                # Execute the Scrapy spider and capture its output
                call_command('run_trends', stdout=out)
                spider_output = out.getvalue()
                
                # Optionally, print or log the output to check for issues
                print("Scrapy Spider Output: ", spider_output)

                # Return the spider output in the JSON response
                return JsonResponse({
                    'spider_output': spider_output.strip(),  # strip to clean up unnecessary whitespace
                })

            except Exception as e:
                # Catch any errors and return them in the JSON response
                return JsonResponse({'error': str(e)}, status=500)
            
  
    @staticmethod
    def generate_title_from_keywords(keywords):
        print("TITLE GENERATION STARTED...........")
        if keywords:
            title = f" {', '.join(keywords[:3])} - Your Ultimate Guide"
            
            return title
        else:
            print("error")
    
    @staticmethod
    def get_video_id(url):
    # Extract video ID from the URL using a simpler regex
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
        return match.group(1) if match else None

    @staticmethod
    def fetch_transcript(video_url):
        video_id = YouTubeVideo.get_video_id(video_url)
        print("**********************************************",end="")
        print("VIDEO ID:",video_id)
        print("**********************************************",end="")

        if not video_id:
            raise ValueError("Invalid YouTube URL")

        try:
            # Fetch the transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            print("TRANSCRIPT:",transcript)
            print("**********************************************",end="")

            
            # Combine transcript text into a single string
            full_transcript = ' '.join([entry['text'] for entry in transcript])
            print("FULLTRANSCRIPT:",full_transcript)
            
            # Limit to the first 200 characters
            return full_transcript
        except Exception as e:
            return f"Error fetching transcript: {e}"
    
    @staticmethod
    def generate_keywords_from_google_trends(keyword):
        """
        Fetch related keywords using Google Trends and return them as JSON.
        """
        pytrends = TrendReq(hl='en-US', tz=360)

        try:
            # Build payload for Google Trends
            print(f"DEBUG: Building payload for keyword: {keyword}")
            pytrends.build_payload([keyword], cat=0, timeframe='today 1-m', geo='IN', gprop='')

            # Fetch related queries
            related_queries = pytrends.related_queries()

            # Log the raw response
            print(f"DEBUG: Raw related_queries response: {related_queries}")

            # Check if the keyword exists in the related_queries response
            if keyword not in related_queries:
                print(f"DEBUG: Keyword '{keyword}' not found in related_queries.")
                return {"keywords": []}

            # Check if 'top' is in the response for the keyword
            if 'top' not in related_queries[keyword] or related_queries[keyword]['top'] is None:
                print(f"DEBUG: 'top' data is missing for keyword '{keyword}'.")
                return {"keywords": []}

            # Extract the 'top' data
            queries_data = related_queries[keyword]['top']
            print(f"DEBUG: Extracted 'top' data: {queries_data}")

            # Ensure that the 'query' column exists in the 'top' data
            if 'query' not in queries_data.columns:
                print(f"DEBUG: 'query' column is missing in 'top' data for keyword '{keyword}'.")
                return {"keywords": []}

            # Extract keywords from the 'top' data
            keywords = queries_data['query'].tolist()
            print(f"DEBUG: Extracted keywords for '{keyword}': {keywords}")

            return {"keywords": keywords}


        except KeyError as ke:
            print(f"KeyError in generate_keywords_from_google_trends: {ke}")
            return {"error": f"KeyError: {str(ke)}"}
        except IndexError as ie:
            print(f"IndexError in generate_keywords_from_google_trends: {ie}")
            return {"error": f"IndexError: {str(ie)}"}
        except Exception as e:
            print(f"Unexpected error in generate_keywords_from_google_trends: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

    @staticmethod
    def generate_interest_over_time(keyword):
        from pytrends.request import TrendReq
        import time

        pytrends = TrendReq(hl='en-US', tz=360)
        try:
            print(f"DEBUG: Building payload for keyword: {keyword}")
            pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='IN', gprop='')

            # Add a delay to prevent rate-limiting
            time.sleep(10)

            # Fetch interest over time data
            interest_data = pytrends.interest_over_time()
            print(f"DEBUG: Interest over time data: {interest_data}")

            if interest_data.empty:
                return {"error": "No interest over time data available"}

            # Extract dates and interest levels
            interest_over_time = interest_data[keyword].reset_index().to_dict(orient='records')
            print(f"DEBUG: Processed interest over time data: {interest_over_time}")

            return {"interest_over_time": interest_over_time}
        except Exception as e:
            print(f"Unexpected error in generate_interest_over_time: {e}")
            return {"error": f"Unexpected error: {str(e)}"}


    @staticmethod
    def generate_interest_by_region(keyword):
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl='en-US', tz=360)
        try:
            print(f"DEBUG: Building payload for keyword: {keyword}")
            pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='IN', gprop='')

            # Fetch interest by region data
            region_data = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
            print(f"DEBUG: Interest by region data: {region_data}")

            if region_data is None or region_data.empty:
                return {"error": "No interest by region data available"}

            # Process the data
            region_interest = region_data[keyword].reset_index().to_dict(orient='records')
            print(f"DEBUG: Processed interest by region data: {region_interest}")

            return {"interest_by_region": region_interest}
        except Exception as e:
            print(f"Unexpected error in generate_interest_by_region: {e}")
            return {"error": f"Unexpected error: {str(e)}"}
        

    @staticmethod
    def fetch_keywords_from_apis_new(text):
        # Placeholder for keyword extraction logic
        cached_result = cache.get(text)
        if cached_result:
            keywords = cached_result
            return keywords
        keywords = []

        # YouTube Data API Key
        api_key = API_CONFIG['api_key']  # Make sure to set this in your config.py

        # Search YouTube for relevant keywords
        youtube_url = API_CONFIG['url']
        params = {
            'part': 'snippet',
            'q': text,
            'maxResults': 10,  # Adjust the number of results
            'key': api_key,
        }

        # Check if the keyword is "python" and generate variations
        if text.lower() == "python":
            text = YouTubeVideo.generate_python_keywords()

        # Fetch data from YouTube API
        response = requests.get(youtube_url, params=params)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('items', []):
                keyword = item['snippet']['title']

                # Set a default value if search_volume cannot be retrieved
                search_volume = 0  # or some other default value

                keywords.append((keyword, search_volume))
                Keyword.objects.create(keyword=keyword, search_volume=search_volume, source="YouTube API")
            
                # Cache the result to avoid redundant API calls
                cache.set(text, keywords, timeout=API_CONFIG['cache_timeout'])

        return keywords
    @staticmethod
    def generate_python_keywords():
        # Generate a list of keyword combinations for "python" with a-z and 0-9 prefixes and suffixes
        prefixes = list(string.ascii_lowercase) + list(map(str, range(10)))  # a-z + 0-9
        suffixes = list(string.ascii_lowercase) + list(map(str, range(10)))  # a-z + 0-9

        # Generate all combinations of the form "prefix + 'python' + suffix"
        python_keywords = [f"{prefix}python{suffix}" for prefix in prefixes for suffix in suffixes]
        print("python keywords:",python_keywords)
        return python_keywords   
    
    
    def get_youtube_suggestions(query):
        url = f'https://suggestqueries.google.com/complete/search?client=youtube&hl=en&ds=yt&q={urllib.parse.quote(query)}'

        print("url:", url)
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return []

        # Log the full raw response text for debugging
        print(f"Response for '{query}': {response.text}")

        try:
            # Manually extract the JSON-like portion of the response string
            start_idx = response.text.find('[')  # Find the start of the JSON array
            end_idx = response.text.rfind(']')  # Find the end of the JSON array

            if start_idx == -1 or end_idx == -1:
                print("Failed to find the JSON data in the response.")
                return []

            # Extract the JSON portion and parse it
            json_str = response.text[start_idx:end_idx+1]
            suggestions = json.loads(json_str)

            print(f"Suggestions: {suggestions[1]}")  # Print the suggestions list
            return suggestions[1]  # Extract and return the suggestion list

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response text: {response.text}")  # Log the raw response for debugging
            return []


    @staticmethod
    def generate_suggestions_for_prefixes(base_query):
        # List of alphabets from a to z
        prefixes = 'abcdefghijklmnopqrstuvwxyz'
        
        all_suggestions = {}
        
        # Loop through each prefix and get suggestions
        for prefix in prefixes:
            query = f'{prefix} {base_query}'
            suggestions = YouTubeVideo.get_youtube_suggestions(query)
            all_suggestions[f'{prefix} {base_query}'] = suggestions
            
        return all_suggestions

    @staticmethod
    def get_youtube_video_urls(query):
        try:
            # Search for videos
            video_search = VideosSearch(query, limit=5)
            results = video_search.result()
            print("Results keys:", results.keys())  # Debug to check keys in results
            
            # Extract video URLs
            videos_data = results.get('result', [])  # Adjust based on actual key
            print("Videos data:", videos_data)      # Debug the list of videos

            video_urls = [video['link'] for video in videos_data if 'link' in video]
            print("Video URLs:", video_urls)        # Final extracted URLs
            return video_urls

        except Exception as e:
            print(f"Error: {e}")
            return []

    @staticmethod
    def generate_thumbnails_from_stable_diffusion_old(keyword):
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
        
        # Ensure you replace 'YOUR_API_KEY' with your actual API key
        api_key = "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"
        
        # Prepare the payload
        payload = json.dumps({
            "key": api_key,
            "prompt": keyword,  # Use the keyword as the prompt
            "negative_prompt": "",  # Set this to an empty string if not using negative prompts
            "width": 512,  # Set width for the generated thumbnails
            "height": 512,  # Set height for the generated thumbnails
            "samples": 5,  # Number of images (thumbnails) to generate
            "num_inference_steps": 20,  # Number of inference steps
            "seed": None,  # Optional: Set a seed for reproducibility (or None for random)
            "guidance_scale": 7.5,  # Adjust the scale as necessary
            "safety_checker": "yes",  # Safety filter for the generated images
            "multi_lingual": "no",  # If the API supports multilingual inputs, change accordingly
            "panorama": "no",  # Set to 'yes' for panoramic images
            "self_attention": "no",  # Self-attention parameter (optional)
            "upscale": "no",  # Set to 'yes' if you want upscaled images
            "embeddings_model": None,  # Optional: Specify an embeddings model if needed
            "webhook": None,  # Optional: If you need a webhook for notifications
            "track_id": None  # Optional: You can specify a tracking ID for the request
        })
        
        # Set the request headers
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Make the POST request to the API
        response = requests.post(api_url, headers=headers, data=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()  # Parse the response as JSON
            thumbnails = data.get("images", [])  # Extract the images from the response
            
            if thumbnails:
                print("Thumbnails generated successfully.")
                return thumbnails  # Return the list of generated thumbnails
            else:
                print("No images returned in the response.")
                return []  # Return an empty list if no images are returned
        else:
            # Log an error if the request failed
            print(f"Failed to generate thumbnails: {response.status_code}")
            print(response.text)  # Print the error message from the API
            return []  # Return an empty list if failed
        

  
    @staticmethod
    def generate_autochapters_from_transcription(transcription):
        """
        Generate autochapters based on transcription text.
        """
        try:
            # Split the transcription into sentences or paragraphs
            sentences = transcription.split('. ')
            chapters = []
            current_time = 0  # Starting timestamp (in seconds)
            chapter_length = 3600  # Approx. duration of each chapter (in seconds)

            # Generate chapter titles and timestamps
            for i, sentence in enumerate(sentences):
                # Assume each sentence contributes equally to time progression
                chapter_start_time = timedelta(seconds=current_time)
                chapter_title = f"Chapter {i + 1}: {sentence[:100]}"  # Title limited to 50 chars
                chapters.append({
                    "start_time": str(chapter_start_time),  # Format: HH:MM:SS
                    "title": chapter_title
                })
                current_time += chapter_length

            return chapters

        except Exception as e:
            logger.error("Error generating autochapters: %s", str(e))
            return []


    @staticmethod
    def generate_thumbnails_from_stable_diffusion(text_prompt, num_images=5):
        print("Text prompt....:", text_prompt)
        
        # Stable Diffusion API URL on Hugging Face
        stable_diffusion_url = STABLEDIFFUSION_API_CONFIG['url']

        headers = {
            "Authorization": f"Bearer {STABLEDIFFUSION_API_CONFIG['api_key']}"  # Your Hugging Face API token
        }
        thumbnails = []
        save_directory = "media/thumbnails/"
        
        # Ensure the save directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Loop to generate the specified number of images
        
        for i in range(num_images):
            varied_prompt = f"{text_prompt} - version {i+1}"
            data = {
                "inputs": varied_prompt,
                "options": {"wait_for_model": True},
            }

            # Make the API call to generate the image
            response = requests.post(stable_diffusion_url, headers=headers, json=data)

            # Debugging: Print out the response status code
            print(f"Response Status Code: {response.status_code}")

            # Check if response is JSON or binary
            content_type = response.headers.get('Content-Type', '')
            print("Content-Type:", content_type)

            if response.status_code == 200:
                if 'application/json' in content_type:
                    try:
                        response_data = response.json()  # Parse as JSON
                        print("API Response (JSON):", response_data)  # Print the entire JSON response

                        # Handle the response as JSON with 'generated_images'
                        if 'generated_images' in response_data:
                            image_data = response_data['generated_images'][0]  # First image

                            # Base64 encoded image handling
                            if image_data.startswith("data:image"):
                                image_base64 = image_data.split(",")[1]  # Extract base64 string
                                image_bytes = base64.b64decode(image_base64)
                                image = Image.open(BytesIO(image_bytes))

                                # Save the image and append the path to thumbnails list
                                thumbnail_path = os.path.join(save_directory, f"{uuid.uuid4()}.png")
                                image.save(thumbnail_path)
                                thumbnails.append(thumbnail_path)
                            else:
                                print("Received image URL:", image_data)
                                thumbnails.append(image_data)  # Append image URL if available
                        
                        else:
                            print("Error: No 'generated_images' field found in response.")
                    
                    except Exception as e:
                        print(f"Error parsing JSON response: {e}")
                elif 'image' in content_type:
                    # Handle binary image data
                    try:
                        image = Image.open(BytesIO(response.content))
                        thumbnail_path = os.path.join(save_directory, f"{uuid.uuid4()}.png")
                        image.save(thumbnail_path)
                        print(f"Thumbnail saved to: {thumbnail_path}")
                        thumbnails.append(thumbnail_path)
                    except Exception as e:
                        print(f"Error processing binary image data: {e}")
                else:
                    print("Unexpected content type:", content_type)

            elif response.status_code == 429:
                print("Rate limit reached; waiting to retry...")
                time.sleep(60)  # Wait a minute before retrying        
            else:
                print(f"API request failed with status code {response.status_code}")
                print(f"Error Response: {response.text}")
        
        # Return the list of all generated thumbnails after the loop completes
        return thumbnails
        



@csrf_exempt    
def extract_keywords(request):
    if request.method == 'POST':
        video_url = request.POST.get('video_url')
        video_file = request.FILES.get('video_file')
        text = request.POST.get("text")
        

        if not video_url and not video_file and not text:
            return JsonResponse({'error': 'No URL , video file or keywords provided'}, status=400)

        if video_url:
            try:
                video = YouTubeVideo(video_url)
                keywords = video.fetch_video_details()
                print(keywords)
                return JsonResponse({
                    'video_id': video.video_id,
                    'title': video.title,
                    'description': video.description,
                    'tags': video.tags,
                    'keywords': keywords
                })
            except Exception as e:
                logger.error("Error fetching video details: %s", str(e))
                return JsonResponse({'error': str(e)}, status=500)

        if video_file:
            logger.info("Processing uploaded video file...")
            try:
                audio_file_path = YouTubeVideo.process_video_file(video_file)
                logger.info("Audio file path: %s", audio_file_path)

                logger.info("About to transcribe audio from: %s", audio_file_path)
                transcription = YouTubeVideo.transcribe_audio(audio_file_path)
                print("Transcription:",transcription)
                logger.info("Transcription completed.")

                # Generate keywords from transcription
                keywords_with_volume = YouTubeVideo.fetch_keywords_from_apis(transcription)
                keywords = [kw[0] for kw in keywords_with_volume]
                print("Keywords are:...",keywords)
                

                #generate title
                title=YouTubeVideo.generate_title_from_keywords(keywords)
                #title =YouTubeVideo.generate_multiple_sentence_titles(basekeywords,max_titles=9)
                print("title:",title)
                #generate description    
                description=YouTubeVideo.generate_description_from_keywords(keywords)
                print("description",description)

                #generate tags
                tags = keywords[:5]
             

                return JsonResponse(
                    {
                        'keywords': keywords,
                        'title': title,
                        'description':description,
                        'tags':tags
                        
                    }
                ) 
    
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)
        if text:
            
        # Generate keywords from transcription
            keywords_with_volume = YouTubeVideo.fetch_keywords_from_apis(text)
            keywords = [kw[0] for kw in keywords_with_volume]
            print("Keywords are:...",keywords)
                
            #generate title
            title=YouTubeVideo.generate_title_from_keywords(keywords)
            #title =YouTubeVideo.generate_multiple_sentence_titles(basekeywords,max_titles=9)
            print("title:",title)
            #generate description    
            description=YouTubeVideo.generate_description_from_keywords(keywords)
            print("description",description)

            #generate tags
            tags = keywords[:5]
             

            return JsonResponse(
                {
                    'keywords': keywords,
                    'title': title,
                    'description':description,
                    'tags':tags
                        
                }
            ) 
    
            
    return JsonResponse({'error': 'Invalid request method'}, status=400)
from datetime import datetime
@csrf_exempt
def home(request):
    referral_code = request.GET.get('referral_code')  # Get the referral code from URL

    # Find affiliate user based on referral code if available
    if referral_code:
        try:
            affiliate_user = AffiliateUser.objects.get(referral_code=referral_code)
            # Optionally log the referral usage or update login time
            affiliate_user.user_logged_in_date = datetime.now()  # Update the login time
            affiliate_user.save()
            email = affiliate_user.email  # Use email for prepopulation
        except AffiliateUser.DoesNotExist:
            email = ''
    else:
        email = ''

    return render(request, 'home.html',{'email': email,'referral_code':referral_code})
def results(request):
    return render(request, 'results.html')
def exraction(request):
    return render(request,'extraction.html')
@csrf_exempt
def extract_keywordss(request):
    if request.method=="POST":
        text=request.POST.get("text")
        # Generate keywords from transcription
        keywords_with_volume = YouTubeVideo.fetch_keywords_from_apis(text)
        keywords = [kw[0] for kw in keywords_with_volume]
        print("Keywords are:...",keywords)

    return render(request,'results.html',{'keywords':keywords})  

@csrf_exempt
def youtube_url(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    if 'file' not in request.FILES:
        return JsonResponse({"error": "No file provided in request"}, status=400)

    video_file = request.FILES['file']
    try:
        # Save and process the video file
        video_file_path = YouTubeVideo.process_video_file(video_file)

        # Transcribe audio and obtain the transcription text
        transcription = YouTubeVideo.transcribe_audio(video_file_path)
        if not transcription:
            return JsonResponse({"error": "Transcription is empty"})

        # Generate keywords from transcription
        keywords_with_volume = YouTubeVideo.fetch_keywords_from_apis(transcription)
        keywords = [kw[0] for kw in keywords_with_volume]
        if not keywords:
            return JsonResponse({"error": "Failed to generate keywords"})

        # Generate title
        title = YouTubeVideo.generate_title_from_keywords(keywords)

        # Generate description
        description = YouTubeVideo.generate_description_from_keywords(keywords)

        # Generate tags (limited to top 5 keywords)
        tags = keywords[:5]

        # Clean up by removing the video file after processing
        os.remove(video_file_path)

        # Prepare and return a success response
        response = {
            "message": "SEO process completed!",
            "description": description,
            "title": title,
            "tags": tags,
        }
        return JsonResponse(response)
    
    except ValueError as e:
        # Handle transcription-related errors, including language errors
        return JsonResponse({"error": str(e)}, status=500)  # Return the exact error message
    
    except Exception as e:
        # Log and handle exceptions
        return JsonResponse({"error": f"Error processing file: {str(e)}"})



@csrf_exempt
def extraction_from_video_old1(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video_file')
        if not video_file:
            return JsonResponse({'error':'Please choose a file'}, status=400)
        # Get the base file name without the unique suffix
        base_file_name = os.path.basename(video_file.name)
        #check if the video file exist in databse
        existing_video_file=videoSEODB.objects.filter(base_file_name=base_file_name).first()
        if existing_video_file:
            print("EXISTS.............")
            #if exists retrieve and return existing data
            title = existing_video_file.title
            description = existing_video_file.description
            tags = existing_video_file.tags
            # thumbnails = YouTubeVideo.generate_thumbnails_from_stable_diffusion(keywords[0])
            
           

            return render(request, 'seo_results.html', {
                'title': title,
                'description': description,
                'tags': tags,
                
                
              
            })
        else:
            # If it doesn't exist, process the video file and generate new data
            try:
                audio_file_path = YouTubeVideo.process_video_file(video_file)
                logger.info("Audio file path: %s", audio_file_path)

                logger.info("About to transcribe audio from: %s", audio_file_path)
                transcription = YouTubeVideo.transcribe_audio(audio_file_path)
                print("Transcription:",transcription)
                logger.info("Transcription completed.")

                # Generate keywords from transcription
                keywords_with_volume = YouTubeVideo.fetch_keywords_from_apis(transcription)
                
                keywords = [kw[0] for kw in keywords_with_volume]
                print("Keywords are:...",keywords)
                

                #generate title
                title=YouTubeVideo.generate_title_from_keywords(keywords)
                #title =YouTubeVideo.generate_multiple_sentence_titles(basekeywords,max_titles=9)
                print("title:",title)
                #generate description    
                description=YouTubeVideo.generate_description_from_keywords(keywords)
                print("description",description)

                #generate tags
                tags = keywords[:5]

                obj=videoSEODB(video_file=video_file, base_file_name=base_file_name,title=title,description=description,tags=tags,keywords=keywords)
                obj.save()

                return render(request,'seo_results.html',
                    {
                       
                        'title': title,
                        'description':description,
                        'tags':tags
                        
                    }
                ) 
    
            except Exception as e:
                return render(request,'seooptions.html')
            

@csrf_exempt
def extraction_from_video_old1(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video_file')
        if not video_file:
            return JsonResponse({'error':'Please choose a file'}, status=400)
        
        else:
            # If it exist, process the video file and generate new data
            try:
                audio_file_path = YouTubeVideo.process_video_file(video_file)
                logger.info("Audio file path: %s", audio_file_path)

                logger.info("About to transcribe audio from: %s", audio_file_path)
                transcription = YouTubeVideo.transcribe_audio(audio_file_path)
                print("Transcription:",transcription)
                logger.info("Transcription completed.")

                # Generate keywords from transcription
                keywords_with_volume = YouTubeVideo.fetch_keywords_from_apis(transcription)
                
                keywords = [kw[0] for kw in keywords_with_volume]
                print("Keywords are:...",keywords)
                

                #generate title
                title=YouTubeVideo.generate_title_from_keywords(keywords)
               
                #generate description    
                description=YouTubeVideo.generate_description_from_keywords(keywords)
                print("description",description)

                #generate tags
                tags = keywords[:5]
                # Autochapters generation
                autochapters = YouTubeVideo.generate_autochapters_from_transcription(transcription)
                print("Autochapters:", autochapters)

                

                return render(request,'seo_results.html',
                    {
                       
                        'title': title,
                        'description':description,
                        'tags':tags,
                        'autochapters': autochapters,
                        
                    }
                ) 
    
            except Exception as e:
                return render(request,'seooptions.html')            
            
@csrf_exempt
def extraction_from_url_old(request):      
    if request.method == "POST":
        video_url = request.POST.get('video_url') 

        if not video_url:
            return JsonResponse({'error': 'No URL provided'}, status=400)
        else:
            try:
                # Check if the video_url already exists in the database
                existing_url = urlSEODB.objects.filter(video_url=video_url).first()
                if existing_url:
                    # If it exists, return the existing data
                    return render(request, 'seo_results.html', {
                        'title': existing_url.title,
                        'description': existing_url.description,
                        'tags': existing_url.tags
                    })
                else:
                # If it doesn't exist, process the URL
                    video = YouTubeVideo(video_url)
                    keywords = video.fetch_video_details()
                    print(keywords)
                    obj=urlSEODB(video_url=video_url,title=video.title,description=video.description,tags=video.tags)
                    obj.save()

                    return render(request,'seo_results.html',
                            {
                            
                                'title': video.title,
                                'description':video.description,
                                'tags':video.tags
                                
                            }
                        )                
            except Exception as e:
                return render(request,'seooptions.html')
            
@csrf_exempt
def extraction_from_text_api_old(request):
    if request.method=="POST":
        text = request.POST.get("text")
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        else: 

            
            suggestions = YouTubeVideo.generate_suggestions_for_prefixes(text)

            # Display the suggestions for each prefix
            for query, suggestion_list in suggestions.items():
                print(f"Suggestions for '{query}':")
                for suggestion in suggestion_list:
                    print(f"- {suggestion}")
                print("\n")
            # Generate keywords from transcription
            # keywords_with_volume = YouTubeVideo.fetch_keywords_from_apis(text)
            keywords_with_volume = YouTubeVideo.fetch_keywords_from_apis(text)
            keywords = [kw[0] for kw in keywords_with_volume]
            print("Keywords are:...",keywords)
     
            #generate title
            title=YouTubeVideo.generate_title_from_keywords(keywords)
            #title =YouTubeVideo.generate_multiple_sentence_titles(basekeywords,max_titles=9)
            print("title:",title)
            #generate description    
            description=YouTubeVideo.generate_description_from_keywords(keywords)
            print("description",description)

            #generate tags
            tags = keywords[:5]
            return render(request,'seo_results.html',
                            {
                            
                                'title': title,
                                'description':description,
                                'tags':tags
                                
                            }
                        )                
@csrf_exempt      
def extraction_from_text_api(request):
    logger.info("Request received")
    if request.method == "POST":
        # Get the keyword entered by the user
        text = request.POST.get("text")
        print("text:",text)
        
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)

        # Generate suggestions based on the text entered
        suggestions = YouTubeVideo.generate_suggestions_for_prefixes(text)
        print("suggestions:",suggestions)
        # Display the suggestions for each prefix
        # for query, suggestion_list in suggestions.items():
        #     print(f"Suggestions for '{query}':")
        #     for suggestion in suggestion_list:
        #         print(f"- {suggestion}")
        #     print("\n")
                
        # Extracting all keywords from the data structure
        keywords = []

        for key in suggestions:
            for suggestion in suggestions[key]:
                keywords.append(suggestion[0])
        print("keywords:",keywords)  

         # Call the Stable Diffusion API to generate thumbnails based on the keyword
        thumbnails = YouTubeVideo.generate_thumbnails_from_stable_diffusion(keywords[0])  # Pass the first keyword      
     
        return JsonResponse({"keywords":keywords, "thumbnails": thumbnails})
        
        # # OR render the results in a template (if you're using Django templates)
        # return render(request, 'seo_results.html', {'suggestions': suggestions})




@csrf_exempt
def extraction_from_text_trends(request):
    if request.method == "POST":
        # Get the keyword entered by the user
        text = request.POST.get("text")
        
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        else:
            try:
                 # Create a StringIO object to capture Scrapy output
                out = StringIO()
                # Execute the Scrapy spider and capture its output
                call_command('run_trends', stdout=out)
                spider_output = out.getvalue()
                
                # Optionally, print or log the output to check for issues
                print("Scrapy Spider Output: ", spider_output)

                # Return the spider output in the JSON response
                return JsonResponse({
                    'spider_output': spider_output.strip(),  # strip to clean up unnecessary whitespace
                })

            except Exception as e:
                # Catch any errors and return them in the JSON response
                return JsonResponse({'error': str(e)}, status=500)
            


def extraction_from_text_scrapy(request):
    """
    Extract YouTube video details and keywords for a given search text using Scrapy.

    Args:
        request (HttpRequest): Django request object containing POST data.

    Returns:
        JsonResponse: Scraped data or an error message.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid HTTP method, use POST."}, status=405)

    search_text = request.POST.get("text")
    if not search_text:
        return JsonResponse({"error": "No text provided"}, status=400)

    try:
        # Generate YouTube search URL
        query_params = {"search_query": search_text}
        search_url = f"https://www.youtube.com/results?{urlencode(query_params)}"

        # Fetch YouTube video URLs via helper function
        video_urls = YouTubeVideo.get_youtube_video_urls(search_text)

        if not video_urls:
            return JsonResponse({"error": "No video URLs found for the given search text."}, status=404)

        # Extract paths and filter valid video links
        video_paths = [url.split("youtube.com")[-1] for url in video_urls]
        full_video_urls = [
            f"https://www.youtube.com{path}" for path in video_paths if "/watch?v=" in path
        ]

        if not full_video_urls:
            return JsonResponse({"error": "No valid video paths found."}, status=404)

        

        # Create an instance of YouTubeSpiderRunner for a single URL
        spider_runner = YouTubeSpiderRunner()

        # Get scraped data from the spider for the video URL
        result_using_srapy = spider_runner.run_spider_for_multiple_urls(full_video_urls)


        return JsonResponse(
            {"message": "Scrapy spider ran successfully", "data": result_using_srapy},
            status=200
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
    


@csrf_exempt
def extraction_from_url_api(request):      
    if request.method == "POST":
        video_url = request.POST.get('video_url') 

        if not video_url:
            return JsonResponse({'error': 'No URL provided'}, status=400)
        else:
            try:
                # Check if the video_url already exists in the database
                existing_url = urlSEODB.objects.filter(video_url=video_url).first()
                print("URL",existing_url)
                if existing_url:
                    # If it exists, return the existing data
                    return render(request, 'seo_results.html', {
                        'title': existing_url.title,
                        'description': existing_url.description,
                        'tags': existing_url.tags
                    })

                # Fetch the transcription
                transcription = YouTubeVideo.fetch_transcript(video_url)
                
                print("Transcription:",transcription)

                # Generate keywords from transcription
                keywords_with_volume = YouTubeVideo.fetch_keywords_from_apis(transcription)
                keywords = [kw[0] for kw in keywords_with_volume]
                print("Keywords:",keywords)
               

                # Generate title
                title = YouTubeVideo.generate_title_from_keywords(keywords)
                logger.info("Generated title: %s", title)
                print("title:",title)

                # Generate description    
                description = YouTubeVideo.generate_description_from_keywords(keywords)
                logger.info("Generated description: %s", description)
                print("description",description)

                # Generate tags
                tags = keywords[:5]
                logger.info("Generated tags: %s", tags)

                # Save to database
                obj = urlSEODB(video_url=video_url, title=title, description=description, tags=tags)
                obj.save()  # Save the object to the database
                logger.info("Saved object: %s", obj)

                # Return rendered response
                return render(request, 'seo_results.html', {
                    'title': title,
                    'description': description,
                    'tags': tags
                })   

            except Exception as e:
                logger.error("Error occurred: %s", e)
                return render(request, 'seooptions.html', {'error': str(e)})





def extraction_from_url_using_scrapy(request):
    """
    Extract data from a YouTube video URL using a Scrapy spider.

    Args:
        request: Django HTTP request object containing a POST request with a video_url.

    Returns:
        JsonResponse: Contains a success message with scraped data or an error message.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid HTTP method, use POST."}, status=405)

    video_url = request.POST.get("video_url")
    if not video_url:
        return JsonResponse({"error": "No URL provided"}, status=400)

    try:
        # Validate that the URL is a valid YouTube URL
        if "youtube.com/watch?v=" not in video_url:
            return JsonResponse({"error": "Invalid YouTube URL"}, status=400)

       # Create an instance of YouTubeSpiderRunner for a single URL
        spider_runner = YouTubeSpiderRunner()

        # Get scraped data from the spider for the video URL
        result = spider_runner.run_spider_for_single_url(video_url)


        # If no data is returned, provide a user-friendly error
        if not result:
            return JsonResponse({"error": "No data extracted from the provided URL."}, status=404)

        return JsonResponse(
            {"message": "Scrapy spider ran successfully", "data": result}, status=200
        )
    except Exception as e:
        # Capture unexpected errors for better debugging
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)



def get_related_keywords_trends(request):
    """
    Fetch related keywords based on user input using pytrends.
    """
    if request.method == "POST":
        try:
            # Get keywords from the POST request
            keywords_input = request.POST.get("text", "")
            
            # Split the keywords by commas and clean extra whitespace
            main_keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
            
            # Check if keywords are provided
            if not main_keywords:
                return JsonResponse({'status': 'error', 'message': 'No keywords provided.'}, status=400)
            
            # Initialize pytrends
            pytrend = TrendReq(hl='en-US', tz=360)

            # Dictionary to store related keywords
            related_keywords = {}

            for keyword in main_keywords:
                # Get suggestions for the keyword
                suggestions = pytrend.suggestions(keyword=keyword)
                related_keywords[keyword] = [s["title"] for s in suggestions]  # Extract titles only

            return JsonResponse({'status': 'success', 'data': related_keywords}, status=200)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)


def youtube_authenticate_for_affiliated(request,contact_id):
    # Initiate OAuth flow with YouTube scope
    print(f"Authenticating YouTube for contact ID: {contact_id}")
    print("CREDENTIALS_PATH: ", CREDENTIALS_PATH)
    if not os.path.exists(CREDENTIALS_PATH):
        print("Error: credentials file not found.")
        return redirect("error_page")

    flow = Flow.from_client_secrets_file(
        CREDENTIALS_PATH,
        scopes=SCOPES,
        # redirect_uri="https://localhost:8000/auth/callback"
        redirect_uri="https://ada3-103-161-55-72.ngrok-free.app/auth/callback"

    )
    # Get the authorization URL and state
    authorization_url, state = flow.authorization_url(prompt="consent")

    # Store flow and contact_id in the session for later use
    request.session['auth_state'] = state
    request.session['contact_id'] = contact_id

    return redirect(authorization_url)

def callback_affiliated(request):
    try:
        # Log the incoming request data
        print("Request Parameters:", request.GET)

        # Initialize OAuth flow
        flow = Flow.from_client_secrets_file(
            CREDENTIALS_PATH,
            scopes=SCOPES,
            # redirect_uri="https://localhost:8000/auth/callback"
            redirect_uri="https://ada3-103-161-55-72.ngrok-free.app/auth/callback"

        )
        state = request.session.get('auth_state')
        contact_id = request.session.get('contact_id')

        if not contact_id:
            print("Error: Missing contact_id in session.")
            return redirect("error_page")

        # Fetch token
        flow.fetch_token(authorization_response=request.build_absolute_uri(), state=state)

        # Use credentials to access YouTube API
        credentials = flow.credentials
        youtube_service = build("youtube", "v3", credentials=credentials)
        response = youtube_service.channels().list(part="snippet", mine=True).execute()

        # Check for valid response
        if not response.get("items"):
            return JsonResponse({"error": "No channel found for this account"}, status=404)

        channel_info = response["items"][0]["snippet"]
        username = channel_info.get("title")
       
        channel_name = channel_info.get("title")
        # Retrieve the email address from the Google UserInfo API
        userinfo_endpoint = USERINFO_ENDPOINT
        # response = requests.get(USERINFO_ENDPOINT, headers=headers)
        headers = {"Authorization": f"Bearer {credentials.token}"}
        userinfo_response = requests.get(userinfo_endpoint, headers=headers)
        userinfo_data = userinfo_response.json()

        email = userinfo_data.get("email")

        # Store in session
        request.session['channel_name'] = channel_info.get("title")
        request.session['channel_description'] = channel_info.get("description")
        request.session['email'] = userinfo_data.get("email")

        # Check if the user already exists
        existing_user = YouTubeUser.objects.filter(email=email).first()
        if existing_user:
            # Optionally update existing user details if needed
            existing_user.username = username
            existing_user.channel_name = channel_name
            existing_user.save()
        else:
            # Create a new user if not exists
            YouTubeUser.objects.create(username=username, channel_name=channel_name, email=email)
        # Update authentication_flag for the associated contact
        contact = get_object_or_404(ContactForm, id=contact_id)
        contact.authentication_flag = True
        contact.save()

        # Clear sensitive session data
        del request.session['auth_state']
        del request.session['contact_id']

        # Redirect to onboarding
        return redirect('mobilenumber')
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error: {str(e)}")
        return JsonResponse({"error": "Failed to retrieve user info"}, status=500)

    except Exception as e:
        # Log the error
        print(f"Error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)




@csrf_exempt
def youtube_authenticate(request):
    referral_code = request.GET.get('referral_code')  # Get the referral code from URL
    if referral_code:
        request.session['referral_code'] = referral_code

    # Initiate OAuth flow with YouTube scope
    
    print("CREDENTIALS_PATH: ", CREDENTIALS_PATH)
    if not os.path.exists(CREDENTIALS_PATH):
        print("Error: credentials file not found.")
        return redirect("error_page")

    flow = Flow.from_client_secrets_file(
        CREDENTIALS_PATH,
        scopes=SCOPES,
        # redirect_uri="https://localhost:8000/auth/callback"
        redirect_uri="https://ada3-103-161-55-72.ngrok-free.app/auth/callback"

    )
    # Get the authorization URL and state
    authorization_url, state = flow.authorization_url(prompt="consent")

    # Store flow and contact_id in the session for later use
    request.session['auth_state'] = state
    request.session.modified = True  # Mark session as modified
    request.session.save() 
    # Log session data
    print(f"Session data after saving: {request.session.items()}")
    # Log the session key
    print(f"Session Key: {request.session.session_key}")

    print(f"Session Cookies: {request.COOKIES}")


    return redirect(authorization_url)

@csrf_exempt
def callback(request):
    try:
        
        # Log the incoming request data
        print("Request Parameters:", request.GET)

       
        # Log the session key
        print(f"Session Key: {request.session.session_key}")

        # Log session data
        print(f"Session data in callback: {request.session.items()}")
        print("Session before retrieving auth_state:", dict(request.session))

        # Initialize OAuth flow
        flow = Flow.from_client_secrets_file(
            CREDENTIALS_PATH,
            scopes=SCOPES,
            # redirect_uri="https://localhost:8000/auth/callback"
            redirect_uri="https://ada3-103-161-55-72.ngrok-free.app/auth/callback"

        )
        state = request.session.get('auth_state')
        # Retrieve the referral code correctly
        referral_code = request.GET.get('referral_code')
        print("Referral Code:", referral_code)
        if not state:
            print("Error: 'auth_state' is missing from session")
            return JsonResponse({"error": "'auth_state' missing from session"}, status=400)

       
        # Fetch token
        flow.fetch_token(authorization_response=request.build_absolute_uri(), state=state)

        # Use credentials to access YouTube API
        credentials = flow.credentials
        youtube_service = build("youtube", "v3", credentials=credentials)
        response = youtube_service.channels().list(part="snippet", mine=True).execute()

        # Check for valid response
        if not response.get("items"):
            return JsonResponse({"error": "No channel found for this account"}, status=404)

        channel_info = response["items"][0]["snippet"]
        username = channel_info.get("title")
       
        channel_name = channel_info.get("title")
        # Retrieve the email address from the Google UserInfo API
        userinfo_endpoint = USERINFO_ENDPOINT
        # response = requests.get(USERINFO_ENDPOINT, headers=headers)
        headers = {"Authorization": f"Bearer {credentials.token}"}
        userinfo_response = requests.get(userinfo_endpoint, headers=headers)
        userinfo_data = userinfo_response.json()

        email = userinfo_data.get("email")

        # Store in session
        request.session['channel_name'] = channel_info.get("title")
        request.session['channel_description'] = channel_info.get("description")
        request.session['email'] = userinfo_data.get("email")

        # Check if the user already exists
        existing_user = YouTubeUser.objects.filter(email=email).first()
        if existing_user:
            # Optionally update existing user details if needed
            existing_user.username = username
            existing_user.channel_name = channel_name
            existing_user.trial_status = 'Active' 
            existing_user.save()
        else:
            # Create a new user if not exists
            YouTubeUser.objects.create(username=username, channel_name=channel_name, email=email,free_trial_start_date=timezone.now(),trial_status='Active'  )
        
        if referral_code:
            affiliated_user=AffiliateUser.objects.filter(referral_code=referral_code).first()
            if affiliated_user:
                # Optionally update existing user details if needed
                affiliated_user.username = username
                affiliated_user.channel_name = channel_name
                affiliated_user.trial_status = 'Active' 
                affiliated_user.user_email=email
                affiliated_user.free_trial_start_date=timezone.now()
                affiliated_user.save()
            
        # Clear sensitive session data
        del request.session['auth_state']
        

        # Redirect to onboarding
        return redirect('mobilenumber')
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error: {str(e)}")
        return JsonResponse({"error": "Failed to retrieve user info"}, status=500)

    except Exception as e:
        # Log the error
        print(f"Error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)
    
 
    
def check_trial_status(user):
    # Check if free trial has expired
    if user.free_trial_start_date:
        trial_end_date = user.free_trial_start_date + timedelta(days=7)
        if timezone.now() > trial_end_date:
            # Mark trial as expired
            user.trial_status = 'Expired'
            user.save()
            print(f"Trial expired for {user.username}.")
        else:
            print(f"Trial still active for {user.username}.")    




def save_mobile(request):
    if request.method == 'POST':
        phone = request.POST.get('phone')  # Get the phone number from the form
        country_code = request.POST.get('country_code')  # Get the country code
        full_phone_number = f"{country_code} {phone}"  # Combine country code and phone number

        # Assuming the user is logged in and has an associated YouTubeUser
        # user = request.user  # If you have a logged-in user, use that. Modify if using a different user logic.
         # Get email from the session
        email = request.session.get('email')

        if not email:
            return redirect('error_page') 
        # Update the YouTubeUser model with the new mobile number
        youtube_user = YouTubeUser.objects.filter(email=email).first()
        if youtube_user:
            youtube_user.mobile_number = full_phone_number
            youtube_user.save()

        return redirect('onboarding')  # Redirect to the onboarding page or wherever you need
    return redirect('error_page')  # In case the request is not POST
             

    
def onboarding(request):
    # Check if the user is authenticated or if you need to fetch their YouTube data
    channel_name = request.session.get('channel_name')
    # channel_description = request.session.get('channel_description')
    email = request.session.get('email')

    return render(request, 'onboarding.html', {
        'channel_name': channel_name,
        'email': email,
    })        

def onboarding_action(request):
    if request.method == 'POST':
        role = request.POST.get('role')
        channel_name = request.session.get('channel_name')
        email = request.session.get('email')
        mobile_number = request.POST.get('mobile_number')

        if not channel_name or not email:
            return JsonResponse({"error": "User information missing."}, status=400)
        
        # Update the YouTubeUser model with the new mobile number
        youtube_user = YouTubeUser.objects.filter(email=email).first()
        if youtube_user:
            youtube_user.role = role
            youtube_user.save()
        
        # Save to the database
        user, created = YouTubeUser.objects.get_or_create(
            username=channel_name,
            defaults={
                'email': email,
                'role': role
            }
        )
        request.session['role'] = role
        request.session['mobile'] = mobile_number


        # Update role if the user already exists
        if not created:
            user.role = role
            user.save()

        if role == 'creator' or role == 'business':
            # Redirect user to the creator dashboard or page
            return redirect('goal_selection')  # You can define the path to creator dashboard
        
        else:
            return JsonResponse({"error": "Invalid role selected."}, status=400)
    return JsonResponse({"error": "Invalid request."}, status=400)
# @login_required
# Ensure that the user is logged in before accessing views that require authentication. You can use the @login_required decorator to enforce login on the view.
def save_goal_data(request):
    if request.method == "POST":
        goal = request.POST.get('goal')
        channel_name = request.session.get('channel_name')
        email = request.session.get('email')
        role = request.session.get('role')

        youtube_user = YouTubeUser.objects.filter(email=email).first()
        if youtube_user:
            youtube_user.goal = goal
            youtube_user.save()
        


        if not all([goal, channel_name, email, role]):
            return JsonResponse({"error": "Incomplete data."}, status=400)

        # Save goal in session
        request.session['goal'] = goal

        # Update or create YouTubeUser entry in the database
        user, created = YouTubeUser.objects.get_or_create(
            
            email=email,
            defaults={
                'channel_name': channel_name,
                'role': role,
                'goal': goal,
            }
        )

        # Update goal if the user already exists
        if not created:
            user.goal = goal
            user.save()

        return redirect('discovery_selection')
    return JsonResponse({"error": "Invalid request."}, status=400)    

def save_discovery_data(request):
    if request.method == "POST":
        discovery = request.POST.get('discovery')
        channel_name = request.session.get('channel_name')
        email = request.session.get('email')
        role = request.session.get('role')
        goal = request.session.get('goal')

        youtube_user = YouTubeUser.objects.filter(email=email).first()
        if youtube_user:
            youtube_user.discovery = discovery
            youtube_user.save()

        if not all([goal, channel_name, email, role,discovery]):
            return JsonResponse({"error": "Incomplete data."}, status=400)

        # Save discovery in session
        request.session['discovery'] = discovery

        # Update or create YouTubeUser entry in the database
        user, created = YouTubeUser.objects.get_or_create(
            
            email=email,
            defaults={
                'channel_name': channel_name,
                'role': role,
                'goal': goal,
                'discovery':discovery
            }
        )

        # Update goal if the user already exists
        if not created:
            user.discovery = discovery
            user.save()

        return redirect('paymentselection')
    return JsonResponse({"error": "Invalid request."}, status=400)  

import json
from decimal import Decimal
from django.http import JsonResponse
from django.utils import timezone
import razorpay

# Razorpay Client Initialization
razorpay_client = razorpay.Client(auth=(RAZORPAY_API_KEY, RAZORPAY_API_SECRET))

from django.http import JsonResponse
from django.shortcuts import render
from .models import PaymentPlan  # Import your model
from razorpay.errors import BadRequestError, ServerError

def update_payment_details_old3(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print(f"Received data: {json.dumps(data, indent=2)}")  # Log the incoming data

            event = data.get('event')
            if not event:
                return JsonResponse({"success": False, "error": "Event is missing or not supported."})

            # Check for event type
            if event == "payment.captured":
                process_successful_payment(data)
            elif event == "payment.failed":
                process_failed_payment(data)
            elif event == "payment.created":
                try:
                    # Validate required fields
                    payment_plan = data.get("payment_plan")
                    payment_term = data.get("payment_term")
                    amount = data.get("amount")
                    currency = data.get("currency")
                    email = request.session.get("email")

                    if not all([payment_plan, payment_term, amount, currency, email]):
                        return JsonResponse({
                            "success": False,
                            "error": "Missing required fields for payment initialization."
                        })

                    # Retrieve or create the user
                    youtube_user, created = YouTubeUser.objects.get_or_create(email=email)

                    # Update user payment details
                    youtube_user.payment_term = payment_term
                    youtube_user.payment_plan = payment_plan
                    youtube_user.amount = amount
                    youtube_user.currency = currency
                    youtube_user.save()

                    logger.info(f"Updated payment details for user: {youtube_user.email}")

                   
                
                    # Retrieve the selected payment plan
                    plan = PaymentPlan.objects.filter(name=payment_plan).first()
                    if not plan:
                        return JsonResponse({"success": False, "error": "Payment plan not found."})

                    # Create Razorpay subscription
                    subscription_data = {
                        "plan_id": plan.plan_id,
                        "total_count": 12 if payment_term.lower() == "monthly" else 1,
                        "customer_notify": 1,
                    }

                    try:
                        subscription = razorpay_client.subscription.create(subscription_data)
                    except (BadRequestError, ServerError) as e:
                        return JsonResponse({"success": False, "error": f"Razorpay Error: {str(e)}"})
                    except Exception as e:
                        return JsonResponse({"success": False, "error": f"An error occurred: {str(e)}"})

                    # Create or update subscription record
                    subscription_record = Subscription_Data.objects.create(
                        youtube_user=youtube_user,
                        plan_name=plan.name,
                        start_date=timezone.now(),
                        end_date=timezone.now() + timezone.timedelta(days=365) if payment_term.lower() == "yearly" else timezone.now() + timezone.timedelta(days=30),
                        razorpay_subscription_id=subscription["id"],
                    )

                    # Create Payment record
                    Payment.objects.create(
                        youtube_user=youtube_user,
                        payment_id=subscription["id"],
                        subscription=subscription_record,
                        amount=amount,
                        currency=currency,
                        status="success",
                        payment_date=timezone.now(),
                        razorpay_order_id=subscription["id"],
                        razorpay_signature=subscription.get("signature", ""),
                    )

                    return JsonResponse({"success": True, 
                                         "subscription_id": subscription["id"],
                                        "message": "Payment details successfully initialized.",
                                        "user": youtube_user.email
                                         })

                except Exception as e:
                    logger.error(f"Error processing payment.created event: {str(e)}")
                    return JsonResponse({"success": False, "error": f"Error: {str(e)}"})

               
            else:
                print(f"Unsupported event: {event}")
                return JsonResponse({"success": False, "error": f"Unsupported event: {event}"})

            
          

            
            
        except Exception as e:
            logger.error(f"Error in update_payment_details: {str(e)}")
            return JsonResponse({"success": False, "error": f"Error: {str(e)}"})
    return JsonResponse({"success": False, "error": "Invalid request method."})
def update_payment_details(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print(f"Received data: {json.dumps(data, indent=2)}")  # Log the incoming data

            event = data.get('event')
            if not event:
                return JsonResponse({"success": False, "error": "Event is missing or not supported."})

            # Check for event type
            if event == "payment.captured":
                process_successful_payment(data)
            elif event == "payment.failed":
                process_failed_payment(data)
            elif event == "payment.created":
                try:
                    # Validate required fields
                    payment_plan = data.get("payment_plan")
                    payment_term = data.get("payment_term")
                    amount = data.get("amount")
                    currency = data.get("currency")
                    email = request.session.get("email")

                    if not all([payment_plan, payment_term, amount, currency, email]):
                        return JsonResponse({
                            "success": False,
                            "error": "Missing required fields for payment initialization."
                        })

                    # Retrieve or create the user
                    youtube_user, created = YouTubeUser.objects.get_or_create(email=email)

                    # Update user payment details
                    youtube_user.payment_term = payment_term
                    youtube_user.payment_plan = payment_plan
                    youtube_user.amount = amount
                    youtube_user.currency = currency
                    youtube_user.save()

                    logger.info(f"Updated payment details for user: {youtube_user.email}")

                    # Retrieve the selected payment plan
                    plan = PaymentPlan.objects.filter(name=payment_plan).first()
                    if not plan:
                        return JsonResponse({"success": False, "error": "Payment plan not found."})

                    # Create Razorpay subscription
                    subscription_data = {
                        "plan_id": plan.plan_id,
                        "total_count": 12 if payment_term.lower() == "monthly" else 1,
                        "customer_notify": 1,
                    }

                    try:
                        subscription = razorpay_client.subscription.create(subscription_data)
                    except (BadRequestError, ServerError) as e:
                        return JsonResponse({"success": False, "error": f"Razorpay Error: {str(e)}"})
                    except Exception as e:
                        return JsonResponse({"success": False, "error": f"An error occurred: {str(e)}"})

                    # Create or update subscription record
                    subscription_record = Subscription_Data.objects.create(
                        youtube_user=youtube_user,
                        plan_name=plan.name,
                        start_date=timezone.now(),
                        end_date=timezone.now() + timezone.timedelta(days=365) if payment_term.lower() == "yearly" else timezone.now() + timezone.timedelta(days=30),
                        razorpay_subscription_id=subscription["id"],
                    )

                    # Create Payment record
                    Payment.objects.create(
                        youtube_user=youtube_user,
                        payment_id=subscription["id"],
                        subscription=subscription_record,
                        amount=amount,
                        currency=currency,
                        status="pending",  # Set status as pending until payment is confirmed
                        payment_date=timezone.now(),
                        razorpay_order_id=subscription["id"],
                        razorpay_signature=subscription.get("signature", ""),
                    )

                    # Generate subscription message
                    payment_link = subscription.get("short_url", "")
                    subscription_message = generate_subscription_message(plan.name, payment_link, "Fozato")
                    print(f"Generated subscription message: {subscription_message}")

                    return JsonResponse({
                        "success": True,
                        "subscription_id": subscription["id"],
                        "message": "Payment details successfully initialized.",
                        "subscription_message": subscription_message,
                        "user": youtube_user.email
                    })

                except Exception as e:
                    logger.error(f"Error processing payment.created event: {str(e)}")
                    return JsonResponse({"success": False, "error": f"Error: {str(e)}"})

            else:
                print(f"Unsupported event: {event}")
                return JsonResponse({"success": False, "error": f"Unsupported event: {event}"})

        except Exception as e:
            logger.error(f"Error in update_payment_details: {str(e)}")
            return JsonResponse({"success": False, "error": f"Error: {str(e)}"})

    return JsonResponse({"success": False, "error": "Invalid request method."})

def generate_subscription_message(plan_name, link, company_name="Your Company Name"):
    """
    Generate a subscription message for the customer.

    :param plan_name: Name of the subscription plan
    :param link: Razorpay payment link
    :param company_name: Your company name (default: "Your Company Name")
    :return: Formatted message
    """
    message = f"""
    Dear Customer,

    Your subscription for the **{plan_name}** has been created but is awaiting the first payment.
    Please complete the payment using the following link: [{link}]({link}).

    If you encounter any issues, feel free to contact our support team.

    Regards,  
    {company_name}
    """
    return message

def process_payment(request):
    if request.method == "POST":
        try:
            # Parse the incoming payment data
            data = json.loads(request.body)
            payment_id = data.get('razorpay_payment_id')
            subscription_id = data.get('razorpay_subscription_id')
            signature = data.get('razorpay_signature')

            # Verify the payment signature using Razorpay's utility
            params_dict = {
                'razorpay_order_id': subscription_id,
                'razorpay_payment_id': payment_id,
                'razorpay_signature': signature,
            }
            try:
                razorpay_client.utility.verify_payment_signature(params_dict)
            except Exception as e:
                return JsonResponse({"success": False, "error": "Payment signature verification failed."})

            # Retrieve the subscription and user
            subscription = Subscription_Data.objects.filter(razorpay_subscription_id=subscription_id).first()
            if not subscription:
                return JsonResponse({"success": False, "error": "Subscription not found."})

            youtube_user = subscription.youtube_user

            # Create or update the payment entry
            Payment.objects.create(
                youtube_user=youtube_user,
                payment_id=payment_id,
                subscription=subscription,
                amount=subscription.youtube_user.amount,  # Assuming subscription amount matches payment
                currency=subscription.youtube_user.currency,
                status="success",
                payment_date=timezone.now(),
                razorpay_order_id=subscription_id,
                razorpay_signature=signature,
            )

            # Update subscription status
            subscription.status = "active"
            subscription.save()

            return JsonResponse({"success": True, "message": "Payment processed successfully."})

        except Exception as e:
            return JsonResponse({"success": False, "error": f"An error occurred: {str(e)}"})

    return JsonResponse({"success": False, "error": "Invalid request method."})

def process_successful_payment(data):
    """
    Process the payment when it is captured successfully.
    This function can include logic like updating the payment status,
    notifying the user, or other post-payment tasks.
    """
    try:
        # Extract relevant data from the incoming payload
        payment_id = data.get('razorpay_payment_id')
        subscription_id = data.get('razorpay_subscription_id')

        # Handle successful payment logic (update subscription status, etc.)
        subscription = Subscription_Data.objects.filter(razorpay_subscription_id=subscription_id).first()
        if subscription:
            subscription.status = "active"
            subscription.save()

        # Log the successful payment or trigger notifications
        print(f"Payment captured successfully: Payment ID - {payment_id}, Subscription ID - {subscription_id}")
        
    except Exception as e:
        print(f"Error processing successful payment: {str(e)}")

def process_failed_payment(data):
    """
    Process the payment when it fails.
    This function can include logic like updating the payment status,
    notifying the user, or other actions when the payment fails.
    """
    try:
        # Extract relevant data from the incoming payload
        payment_id = data.get('razorpay_payment_id')
        subscription_id = data.get('razorpay_subscription_id')

        # Handle failed payment logic (mark subscription as failed, notify user, etc.)
        subscription = Subscription_Data.objects.filter(razorpay_subscription_id=subscription_id).first()
        if subscription:
            subscription.status = "failed"
            subscription.save()

        # Log the failed payment or trigger notifications
        print(f"Payment failed: Payment ID - {payment_id}, Subscription ID - {subscription_id}")
        
    except Exception as e:
        print(f"Error processing failed payment: {str(e)}")


def update_payment_details_old2(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            payment_term = data.get("payment_term")
            payment_plan_name = data.get("payment_plan")
            amount = Decimal(data.get("amount"))
            currency_symbol = data.get("currency")
            channel_name = request.session.get('channel_name')
            email = request.session.get('email')
            role = request.session.get('role')
            goal = request.session.get('goal')
            discovery = request.session.get('discovery')

            # youtube_user_email = data.get("email")  # Assumes email is passed in the request

            
            
            # Check for required fields
            if not all([payment_term, payment_plan_name, amount, currency_symbol]):
                missing_fields = [field for field in ["payment_term", "payment_plan", "amount", "currency","email"] if not data.get(field)]
                if missing_fields:
                    return JsonResponse({"success": False, "error": f"Missing required fields: {', '.join(missing_fields)}"})
                               

            # Retrieve or create YouTubeUser
            youtube_user, created = YouTubeUser.objects.get_or_create(email=email)
            if youtube_user:
                youtube_user.payment_term = payment_term
                youtube_user.payment_plan = payment_plan_name
                youtube_user.amount = amount
                youtube_user.currency = currency_symbol
                youtube_user.save()
                
           
            # Update or create YouTubeUser entry in the database
            else:
                # If the user doesn't exist, create a new one
                youtube_user = YouTubeUser.objects.create(
                    email=email,
                    channel_name=channel_name,
                    role=role,
                    goal=goal,
                    discovery=discovery,
                    payment_term=payment_term,
                    payment_plan=payment_plan_name,
                    amount=amount,
                    currency=currency_symbol
                )
            print("youtube user updated")

            # Retrieve the selected payment plan
            plan = PaymentPlan.objects.filter(name=payment_plan_name).first()
            if not plan:
                return JsonResponse({"success": False, "error": "Payment plan not found."})

            # Log the plan_id to ensure it's correct
            print(f"Retrieved plan: {plan}")

            # Create Razorpay subscription
            subscription_data = {
                "plan_id": plan.plan_id,
                "total_count": 12 if payment_term.lower() == "monthly" else 1,
                "customer_notify": 1,
            }
            print(f"Retrieved plan_id: {plan.plan_id}")
           

            try:
                subscription = razorpay_client.subscription.create(subscription_data)
            except razorpay.errors.BadRequestError as e:
                print(f"Razorpay BadRequestError: {e}")
                return JsonResponse({"success": False, "error": "Bad Request Error. Please check your subscription data."})
            except razorpay.errors.ServerError as e:
                print(f"Razorpay ServerError: {e}")
                return JsonResponse({"success": False, "error": "Server Error. Please try again later."})
            except Exception as e:
                if "Authentication failed" in str(e):
                    print(f"Razorpay Authentication Error: {e}")
                    return JsonResponse({"success": False, "error": "Authentication failed. Check Razorpay API keys."})
                print(f"Razorpay General Error: {e}")
                return JsonResponse({"success": False, "error": f"An error occurred: {str(e)}"})

            # Create or update subscription record in Subscription_Data
            subscription_data = Subscription_Data.objects.create(
                youtube_user=youtube_user,
                plan_name=plan.name,
                start_date=timezone.now(),
                end_date=timezone.now() + timezone.timedelta(days=365) if payment_term.lower() == "yearly" else timezone.now() + timezone.timedelta(days=30),
                razorpay_subscription_id=subscription["id"],
            )
            
            
            
            # Create Payment record
            Payment.objects.create(
                youtube_user=youtube_user,
                payment_id=subscription["id"],
                subscription=subscription_data,
                amount=amount,
                currency=currency_symbol,
                status="success",
                payment_date=timezone.now(),
                razorpay_order_id=subscription["id"],
                razorpay_signature=subscription.get("signature", ""),
            )

            return JsonResponse({"success": True, "subscription_id": subscription["id"]})

        except Exception as e:
            return JsonResponse({"success": False, "error": f"Error: {str(e)}"})
    
    return JsonResponse({"success": False, "error": "Invalid request method."})   

def process_payment_old2(request):
    if request.method == "POST":
        try:
            # Parse the incoming payment data
            data = json.loads(request.body)
            payment_id = data.get('razorpay_payment_id')
            subscription_id = data.get('razorpay_subscription_id')
            signature = data.get('razorpay_signature')

            # Verify the payment signature using Razorpay's utility
            params_dict = {
                'razorpay_order_id': subscription_id,
                'razorpay_payment_id': payment_id,
                'razorpay_signature': signature,
            }
            try:
                razorpay_client.utility.verify_payment_signature(params_dict)
            except Exception as e:
                return JsonResponse({"success": False, "error": "Payment signature verification failed."})

            # Retrieve the subscription and user
            subscription = Subscription_Data.objects.filter(razorpay_subscription_id=subscription_id).first()
            if not subscription:
                return JsonResponse({"success": False, "error": "Subscription not found."})

            # Retrieve or create YouTubeUser
            youtube_user = subscription.youtube_user

            # Create or update the payment entry
            Payment.objects.create(
                youtube_user=youtube_user,
                payment_id=payment_id,
                subscription=subscription,
                amount=subscription.youtube_user.amount,  # Assuming subscription amount matches payment
                currency=subscription.youtube_user.currency,
                status="success",
                payment_date=timezone.now(),
                razorpay_order_id=subscription_id,
                razorpay_signature=signature,
            )

            # Update subscription status
            subscription.status = "active"
            subscription.save()
            

            return JsonResponse({"success": True, "message": "Payment processed successfully."})

        except Exception as e:
            return JsonResponse({"success": False, "error": f"An error occurred: {str(e)}"})

    return JsonResponse({"success": False, "error": "Invalid request method."}) 

@csrf_exempt
def razorpay_webhook(request):
    webhook_secret = 'RMkzX3kh8@kL@9B'  # Secret configured in Razorpay
    signature = request.headers.get('x-razorpay-signature')
    payload = request.body

    # Verify the signature
    try:
        razorpay_client.utility.verify_payment_signature({
            'razorpay_order_id': request.POST['razorpay_order_id'],
            'razorpay_payment_id': request.POST['razorpay_payment_id'],
            'razorpay_signature': signature,
        })
    except razorpay.errors.SignatureVerificationError:
        return JsonResponse({'error': 'Invalid signature'}, status=400)

    # Process the event after successful signature verification
    return JsonResponse({'success': 'Webhook processed'}, status=200)

def update_payment_details_old1(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            payment_term = data.get("payment_term")
            payment_plan_name = data.get("payment_plan")
            amount = Decimal(data.get("amount"))
            currency_symbol = data.get("currency")
            
            # Check for required fields
            if not all([payment_term, payment_plan_name, amount, currency_symbol]):
                return JsonResponse({"success": False, "error": "Missing required fields."})

            # Retrieve the selected payment plan
            plan = PaymentPlan.objects.filter(name=payment_plan_name).first()
            if not plan:
                return JsonResponse({"success": False, "error": "Payment plan not found."})
            # Log the plan_id to ensure it's correct
            print(f"Retrieved plan: {plan}")

            # Create Razorpay subscription
            subscription_data = {
                "plan_id": plan.plan_id,
                "total_count": 12 if payment_term.lower() == "monthly" else 1,
                "customer_notify": 1,
            }
            print(f"Retrieved plan_id: {plan.plan_id}")
            try:
                subscription = razorpay_client.subscription.create(subscription_data)
            except razorpay.errors.BadRequestError as e:  # Handle Razorpay-specific errors
                print(f"Razorpay BadRequestError: {e}")
                return JsonResponse({"success": False, "error": "Bad Request Error. Please check your subscription data."})
            except razorpay.errors.ServerError as e:  # Handle server errors
                print(f"Razorpay ServerError: {e}")
                return JsonResponse({"success": False, "error": "Server Error. Please try again later."})
            except Exception as e:  # Handle generic errors
                if "Authentication failed" in str(e):
                    print(f"Razorpay Authentication Error: {e}")
                    return JsonResponse({"success": False, "error": "Authentication failed. Check Razorpay API keys."})
                print(f"Razorpay General Error: {e}")
                return JsonResponse({"success": False, "error": f"An error occurred: {str(e)}"})


            return JsonResponse({"success": True, "subscription_id": subscription["id"]})
        except Exception as e:
            return JsonResponse({"success": False, "error": f"Error: {str(e)}"})
    return JsonResponse({"success": False, "error": "Invalid request method."})


def process_payment_old(request):
    if request.method == "POST":
        try:
            # Parse the incoming payment data
            data = json.loads(request.body)
            payment_id = data.get('razorpay_payment_id')
            subscription_id = data.get('razorpay_subscription_id')
            signature = data.get('razorpay_signature')

            # Verify the payment signature using Razorpay's utility
            params_dict = {
                'razorpay_order_id': subscription_id,
                'razorpay_payment_id': payment_id,
                'razorpay_signature': signature,
            }
            try:
                Utility.verify_payment_signature(params_dict)
            except Exception as e:
                return JsonResponse({"success": False, "error": "Payment signature verification failed."})

            # Retrieve the subscription and user
            subscription = Subscription_Data.objects.filter(razorpay_subscription_id=subscription_id).first()
            if not subscription:
                return JsonResponse({"success": False, "error": "Subscription not found."})

            # Create or update the payment entry
            Payment.objects.create(
                youtube_user=subscription.youtube_user,
                payment_id=payment_id,
                subscription=subscription,
                amount=subscription.youtube_user.amount,  # Assuming subscription amount matches payment
                currency=subscription.youtube_user.currency,
                status="success",
                payment_date=timezone.now(),
                razorpay_order_id=subscription_id,
                razorpay_signature=signature,
            )

            # Update subscription status if necessary
            subscription.status = "active"
            subscription.save()

            return JsonResponse({"success": True, "message": "Payment processed successfully."})

        except Exception as e:
            return JsonResponse({"success": False, "error": f"An error occurred: {str(e)}"})

    return JsonResponse({"success": False, "error": "Invalid request method."})


def update_payment_details_old(request):
    if request.method == "POST":
        try:
            # Get the data from the AJAX request
            data = json.loads(request.body)
            payment_term = data.get("payment_term")
            payment_plan = data.get("payment_plan")
            amount = data.get("amount")
            currency_symbol = data.get("currency")
            channel_name = request.session.get('channel_name')
            email = request.session.get('email')
            role = request.session.get('role')
            goal = request.session.get('goal')
            discovery = request.session.get('discovery')

            # Retrieve the predefined plan from your database
            predefined_plan = Payment.objects.filter(name=payment_plan).first()
            if not predefined_plan:
                return JsonResponse({"success": False, "error": "Payment plan not found in the database."})

            # # Ensure amount is a valid Decimal value
            # if amount_str:
            #     # Assuming the amount string is in a format like "$12.49" or "₹832"
            #     amount = Decimal(amount_str.replace('$', '').replace('₹', '').strip())
            # else:
            #     raise ValueError("Amount is required and must be a valid number.")

            youtube_user = YouTubeUser.objects.filter(email=email).first()
            if youtube_user:
                youtube_user.payment_term = payment_term
                youtube_user.payment_plan = payment_plan
                youtube_user.amount = amount
                youtube_user.currency = currency_symbol
                youtube_user.save()
                
           
            # Update or create YouTubeUser entry in the database
            else:
                # If the user doesn't exist, create a new one
                youtube_user = YouTubeUser.objects.create(
                    email=email,
                    channel_name=channel_name,
                    role=role,
                    goal=goal,
                    discovery=discovery,
                    payment_term=payment_term,
                    payment_plan=payment_plan,
                    amount=amount,
                    currency=currency_symbol
                )
            print("youtube user updated")    
            # Create Razorpay Plan (this step should ideally be done on Razorpay Dashboard or once globally)
            
            # Use the predefined Razorpay plan_id to create a subscription
            try:
                subscription = razorpay_client.subscription.create({
                    "plan_id": predefined_plan.plan_id,  # Use the stored plan_id
                    "customer_notify": 1,
                    "total_count": None  # Unlimited billing cycles
                })
            except Exception as e:
                print(f"Error in Razorpay subscription creation: {str(e)}")
                return JsonResponse({"success": False, "error": "Subscription creation failed."})
            # Example: Create a trial subscription for the user
            Subscription_Data.objects.create(
                youtube_user=youtube_user,
                plan_name=payment_plan,
                start_date=timezone.now(),
                end_date=timezone.now() + timezone.timedelta(days=30),
                is_active=True,
                razorpay_subscription_id=subscription['id']
            )    
            # Example: Log a dummy payment for the trial (can be updated for actual payments)
            Payment.objects.create(
                youtube_user=youtube_user,
                payment_id="TRIAL_PAYMENT",
                amount=int(amount * 100),
                currency=currency_symbol,
                status="success",
                payment_date=timezone.now(),
                payment_term = payment_term
            )
            
             # Save payment details in session
            request.session['payment_term'] = payment_term  
            request.session['payment_plan'] = payment_plan 
            request.session['amount'] = str(amount)
            request.session['currency'] = currency_symbol
            # Return success response
            return JsonResponse({"success": True, "subscription_id": subscription['id']})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    return JsonResponse({"success": False, "error": "Invalid request method"})  


from django.urls import reverse
@csrf_exempt
def cancel_subscription(request):
    if request.method == "POST":
        try:
            # Parse subscription ID from the request body
            data = json.loads(request.body)
            subscription_id = data.get("subscription_id")

            if not subscription_id:
                return JsonResponse({"success": False, "error": "No subscription ID provided."})

            # Cancel the subscription via Razorpay API
            razorpay_client.subscription.cancel(subscription_id)

            # Update the subscription's status in the database
            subscription = Subscription_Data.objects.get(razorpay_subscription_id=subscription_id)
            subscription.is_active = False
            subscription.status = 'cancelled'
            subscription.save()
             # Redirect to the user dashboard
            user_dashboard_url = reverse('user_dashboard')
            return JsonResponse({"success": True, "message": "Subscription canceled successfully.", "redirect_url": user_dashboard_url})
        except Subscription_Data.DoesNotExist:
            return JsonResponse({"success": False, "error": "Subscription not found."})
        # except Exception as e:
        #     return JsonResponse({"success": False, "error": str(e)})
    return JsonResponse({"success": False, "error": "Invalid request method."})

def check_user_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        print("view function")
        try:
            # Fetch the YouTubeUser
            user = YouTubeUser.objects.get(username=username, email=email)

            # Fetch the active subscription for the user, if it exists
            subscription = Subscription_Data.objects.filter(
                youtube_user=user,
                is_active=True,
                status='active'
            ).first()  # Use .first() to get the first matching subscription or None

            # Store user info in session
            request.session['user_id'] = user.id
            request.session['username'] = user.username
            request.session['email'] = user.email
            request.session['channel_name'] = user.channel_name
            # Check if a subscription exists
            if subscription:
                subscription_id = subscription.razorpay_subscription_id
                print("subscription_id:", subscription_id)
                # Store subscription details in session
                request.session['subscription_id'] = subscription.razorpay_subscription_id
                request.session['subscription_plan'] = subscription.plan_name
            else:
                subscription_id = None
                request.session['subscription_id'] = None
                request.session['subscription_plan'] = None

            # Redirect to the user dashboard
            return render(request,'user_dashboard1.html',{'subscription_id':subscription_id})

        except YouTubeUser.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Invalid credentials'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return redirect('user_dashboard')


# View for logging out
def logout_user(request):
    request.session.clear()  # Clear session
    return redirect(user_dashboard)    

def razorpay_webhook_old(request):
    print("webhook triggered")
    try:
        data = json.loads(request.body)
        event = data.get('event')
        payload = data.get('payload', {})

        if event == "subscription.charged":
            # Update payment success
            subscription_id = payload['subscription']['entity']['id']
            payment_id = payload['payment']['entity']['id']

            subscription = Subscription_Data.objects.get(razorpay_subscription_id=subscription_id)
            Payment.objects.create(
                youtube_user=subscription.youtube_user,
                payment_id=payment_id,
                amount=subscription.youtube_user.amount,
                currency=subscription.youtube_user.currency,
                status="success",
                payment_date=timezone.now()
            )

        elif event == "subscription.cancelled":
            # Mark subscription as inactive
            subscription_id = payload['subscription']['entity']['id']
            subscription = Subscription_Data.objects.get(razorpay_subscription_id=subscription_id)
            subscription.is_active = False
            subscription.save()

        return HttpResponse(status=200)
    except Exception as e:
        return HttpResponse(status=400)
    
def calculate_mrr():
    active_subscriptions = Subscription_Data.objects.filter(is_active=True)
    return sum(subscription.youtube_user.amount for subscription in active_subscriptions if subscription.youtube_user.payment_term == "monthly")

def calculate_arr():
    mrr = calculate_mrr()
    yearly_revenue = sum(
        subscription.youtube_user.amount for subscription in Subscription_Data.objects.filter(is_active=True, youtube_user__payment_term="Yearly")
    )
    return mrr * 12 + yearly_revenue
def calculate_churn_rate():
    total_subscriptions = Subscription_Data.objects.count()
    cancelled_subscriptions = Subscription_Data.objects.filter(is_active=False, updated_at__gte=timezone.now() - timedelta(days=30)).count()
    return (cancelled_subscriptions / total_subscriptions) * 100 if total_subscriptions > 0 else 0


from razorpay.errors import BadRequestError

def verify_razorpay_signature(request):
    data = json.loads(request.body)
    razorpay_payment_id = data.get('razorpay_payment_id')
    razorpay_subscription_id = data.get('razorpay_subscription_id')
    razorpay_signature = data.get('razorpay_signature')

    # Verify the payment signature
    params_dict = {
        'razorpay_order_id': razorpay_subscription_id,
        'razorpay_payment_id': razorpay_payment_id,
        'razorpay_signature': razorpay_signature
    }

    try:
        razorpay_client.utility.verify_payment_signature(params_dict)
    except Exception as e:
        logger.error(f"Payment signature verification failed: {str(e)}")
        return JsonResponse({"success": False, "error": "Payment signature verification failed."})
    
    return None  # No error, signature is valid
    

def payment_page(request):
    return render(request, 'payment.html')

def create_order(request, amount):
    # Create an order for the specified amount
    order_data = {
        'amount': amount * 100,  # Convert amount to smallest unit (e.g., for $19, use 1900 cents)
        'currency': 'USD',
        'payment_capture': '1'
    }
    order = razorpay_client.order.create(order_data)
    return JsonResponse(order)

@csrf_exempt
def payment_success(request):
    # Verify the payment
    if request.method == "POST":
        payment_id = request.POST.get("razorpay_payment_id")
        order_id = request.POST.get("razorpay_order_id")
        signature = request.POST.get("razorpay_signature")

         # Retrieve data from session and request
        channel_name = request.session.get("channel_name")
        email = request.session.get("email")
        role = request.session.get("role")
        goal = request.session.get("goal")
        discovery = request.session.get("discovery")
        amount = request.session.get("amount")
        plan_duration = request.session.get("plan_duration")

        params_dict = {
            'razorpay_order_id': order_id,
            'razorpay_payment_id': payment_id,
            'razorpay_signature': signature
        }

        # Verify signature
        try:
            razorpay_client.utility.verify_payment_signature(params_dict)
             

            with transaction.atomic():
                Subscription.objects.create(
                    channel_name=channel_name,
                    email=email,
                    role=role,
                    goal=goal,
                    discovery=discovery,
                    amount=amount,
                    plan_duration=plan_duration,
                    payment_id=payment_id,
                    order_id=order_id,
                    signature=signature
                )
            return render(request, "success.html")
        except:
            return render(request, "failure.html")      
        
def contactsave(request):
    if request.method == "POST":
        try:
            print("POST Data:", request.POST)  # Debugging output
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            email = request.POST.get('email')
            country_code = request.POST.get('country_code')
            phone = request.POST.get('phone')
            amount = request.POST.get('amount')

            contact = ContactForm.objects.create(
                first_name=first_name,
                last_name=last_name,
                email=email,
                country_code=country_code,
                phone=phone,
                amount=amount,
                authentication_flag=False  # Default to False
            )
            contact.save()
            print(ContactForm.objects.all())
            # Store contact ID in the session for subsequent operations
            request.session['contact_id'] = contact.id

            return JsonResponse({'success': True, 'contact_id': contact.id})

        except Exception as e:
            print("Error:", e)  # Debugging output
            return JsonResponse({'success': False, 'error': str(e)})

    return render(request, 'contact_form.html')

@csrf_exempt
def save_payment_details(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            payment_id = data.get('payment_id')
            order_id = data.get('order_id')
            signature = data.get('signature')

            # Save these details in the database for testing purposes
            payment = PaymentDetails.objects.create(
                payment_id=payment_id,
                order_id=order_id,
                signature=signature
            )
            payment.save()

            # Retrieve contact ID from the session
            contact_id = request.session.get('contact_id')
            print(f"Contact ID in session: {contact_id}")  # Debugging output
            if not contact_id:
                return JsonResponse({'success': False, 'error': 'Missing contact_id in session'})


            # Redirect to youtube_authenticate
            redirect_url = f"/auth/youtube_affiliated/{contact_id}/"
            print(f"Redirecting to: {redirect_url}")  # Debugging output
            return JsonResponse({'success': True, 'redirect_url': redirect_url})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


# model_id = "nvidia/Llama-3_1-Nemotron-51B-Instruct"
# model_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "device_map": "auto"}
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token_id = tokenizer.eos_token_id

# pipeline = transformers.pipeline(
#     "text-generation", 
#     model=model_id, 
#     tokenizer=tokenizer, 
#     max_new_tokens=20, 
#     **model_kwargs
# )
# print(pipeline([{"role": "user", "content": "SEO keywords for python tutorial"}]))





# API_URL = "https://api-inference.huggingface.co/models/facebook/llama-13b"
# API_URL = "https://api-inference.huggingface.co/models/gpt2"
# headers = {"Authorization": "Bearer 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"}


# def query(payload):
#     try:
#         response = requests.post(API_URL, headers=headers, json=payload)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Request failed: {e}")
#         return None

# # Example usage
# result = query({"inputs": "What is Llama?"})
# if result:
#     print(result)
# else:
#     print("Failed to retrieve a response.")



# ************ USING LLAMA 3.2-3B INSTRUCT,PASSING KW OR DESCRIPTION TO LLAMA TO GET ONE KW ****************
# !pip install transformers
# !pip install huggingface_hub
# !pip install torch
# !pip install accelerate
# import os
# os.environ['HF_TOKEN']="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"
# os.environ['HIGGINGFACEHUB_API_TOKEN']="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"
# import torch
# from transformers import pipeline

# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "please give one seo keywords for python programming"},
# ]
# outputs = pipe(
#     messages,
#     max_new_tokens=256,
# )
# print(outputs[0]["generated_text"][-1])



# ***************PASSING COMBINED KEYWORDS FROM SCRAPY,TRENDS AND API TO LLAMA******************

# def integrate_with_llm(unique_keywords):
#     """
#     Pass combined keywords to the LLM pipeline and generate a response.
#     """
#     # Initialize the pipeline
#     model_id = "meta-llama/Llama-3.2-3B-Instruct"
#     pipe = pipeline(
#         "text-generation",
#         model=model_id,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#     )

#     # Format the input messages
#     messages = [
#         {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#         {
#             "role": "user",
#             "content": f"Please suggest some SEO-friendly keywords for the following combined keywords: {', '.join(unique_keywords)}",
#         },
#     ]

#     # Generate the output
#     outputs = pipe(
#         messages,
#         max_new_tokens=256,
#     )

#     # Return the generated text
#     return outputs[0]["generated_text"]

# unique_keywords=["Java", "Effective Java (3rd Edition)", "Data Structures and Algorithms in Python", "Java: The Complete Reference", "Elements of Programming Interviews in Java: The Insiders' Guide", "the java programming language", "a java program", "java programming btech", "c java programming", "c c++ java programming kya hai", "c c++ java programming interview questions and answers", "c programming for java programmers", "d java programming tutorial", "d java programming for beginners", "java in programming", "java in programming language", "f java programming tutorial", "f java programming language", "f java programming for beginners", "g java programming examples", "g java programming tutorial", "h java programming tutorial", "h java programming for beginners", "i java programming good", "i java programming hard", "i java programming important", "java programming", "java programming for beginners", "java programming full course", "java programming in tamil", "java programming language", "java programming fundamentals infosys", "java programming interview questions and answers", "java programming code with harry", "java programming in telugu", "java programming for beginners in telugu", "java programming bca 5th sem", "java programming one shot", "java programming playlist", "java programming tutorial", "java ke program ko run kaise kare", "java ke program", "l java programming tutorial", "l java programming in hindi", "java me program kaise banaye", "m square programming java", "p java programming tutorial", "p java programming for beginners", "q java programming tutorial", "q java programming language", "q java programming in hindi", "r java programming tutorial", "r java programming for beginners", "introduction to java programming in telugu", "introduction to java programming language", "introduction to java programming", "introduction to java programming for beginners", "intro to java programming", "introduction to java programming btech 2nd year", "introduction to java programming and data structures", "welcome to java programming code", "introduction to java programming in tamil", "how to java programming", "fundamentals to java programming class 11", "introduction to java programming class 8", "vs code java programming", "project with java programming", "starting with java programming", "dsa with java programming", "x java programming language", "x java programming tutorial", "x java programming in hindi", "x java programming in tamil", "y java programming tutorial"]
# pirate_response = integrate_with_llm(unique_keywords)
# print("pirate response:",pirate_response)




# ********************GETTING DESCRIPTION USING 500 KW GENERATED FROM TRENDS AND SEED KW****************
# import torch
# from transformers import pipeline
# def integrate_with_llm(unique_keywords,seed_keyword):
#     """
#     Pass combined keywords to the LLM pipeline and generate a response.
#     """
#     # Initialize the pipeline
#     model_id = "meta-llama/Llama-3.2-3B-Instruct"
#     pipe = pipeline(
#         "text-generation",
#         model=model_id,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#     )

#     # Format the input messages
#     messages = [
#         {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#         {
#             "role": "user",
#             "content": f"Please give description using the following combined keywords: {', '.join(unique_keywords)} and seed keyword {seed_keyword}",
#         },
#     ]

#     # Generate the output
#     outputs = pipe(
#         messages,
#         max_new_tokens=256,
#     )

#     # Return the generated text
#     return outputs[0]["generated_text"]

# unique_keywords=["Java", "Effective Java (3rd Edition)", "Data Structures and Algorithms in Python", "Java: The Complete Reference", "Elements of Programming Interviews in Java: The Insiders' Guide", "the java programming language", "a java program", "java programming btech", "c java programming", "c c++ java programming kya hai", "c c++ java programming interview questions and answers", "c programming for java programmers", "d java programming tutorial", "d java programming for beginners", "java in programming", "java in programming language", "f java programming tutorial", "f java programming language", "f java programming for beginners", "g java programming examples", "g java programming tutorial", "h java programming tutorial", "h java programming for beginners", "i java programming good", "i java programming hard", "i java programming important", "java programming", "java programming for beginners", "java programming full course", "java programming in tamil", "java programming language", "java programming fundamentals infosys", "java programming interview questions and answers", "java programming code with harry", "java programming in telugu", "java programming for beginners in telugu", "java programming bca 5th sem", "java programming one shot", "java programming playlist", "java programming tutorial", "java ke program ko run kaise kare", "java ke program", "l java programming tutorial", "l java programming in hindi", "java me program kaise banaye", "m square programming java", "p java programming tutorial", "p java programming for beginners", "q java programming tutorial", "q java programming language", "q java programming in hindi", "r java programming tutorial", "r java programming for beginners", "introduction to java programming in telugu", "introduction to java programming language", "introduction to java programming", "introduction to java programming for beginners", "intro to java programming", "introduction to java programming btech 2nd year", "introduction to java programming and data structures", "welcome to java programming code", "introduction to java programming in tamil", "how to java programming", "fundamentals to java programming class 11", "introduction to java programming class 8", "vs code java programming", "project with java programming", "starting with java programming", "dsa with java programming", "x java programming language", "x java programming tutorial", "x java programming in hindi", "x java programming in tamil", "y java programming tutorial"]
# seed_keyword="java programming"
# Description = integrate_with_llm(unique_keywords,seed_keyword)
# print("Description:",Description)




