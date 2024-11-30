API_CONFIG = {
    'api_key': 'AIzaSyBPA3t8c4dWtwmXvm4JWa4tyXEYXNDtvo8',
    'url':'https://www.googleapis.com/youtube/v3/search',
    'cache_timeout': 60 * 60 , # 1 hour
    #'cache_enabled': 1  # 1 for enabled, 0 for disabled
    

}
WHISPER_API_CONFIG = {
    'url': 'https://api.whisper.ai/whisper/v1/transcribe',
    'api_key': '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',  # Replace with your actual API key
}
GROK_API_CONFIG  ={
    'url':'https://api.openai.com/v1/chat/completions',
    'api_key': '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'
}


RAZORPAY_API_KEY = "rzp_test_0XYpYSiR8V6hyh"
RAZORPAY_API_SECRET = "toACX9h93ixCjX11NBFNCaHn"

DATAMUSE_API_URL = "https://api.datamuse.com/sug?s={keyword}&max=5"
DATAMUSE_RELATED_URL = "https://api.datamuse.com/words?rel_trg={keyword}&max=10"


KEYWORD_API_BASE_URL = "http://api.example.com/keywords?text={text}"

USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v1/userinfo"

PYTRENDS_DEFAULT_REGION = 'US'  # Default region for Google Trends

STABLEDIFFUSION_API_CONFIG = {
    'api_key':'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',
    'url':'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large'
}


