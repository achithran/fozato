from django.urls import path
from .views import extract_keywords,indexpage,results,exraction,extract_keywordss,index_page,seo_options,extraction_from_video,extraction_from_text,home
from . import views

urlpatterns = [
    path('extract-keywords/', extract_keywords, name='extract_keywords'),
    path('index/',indexpage,name='index'),
    path('index_page/',index_page,name='index_page'),
    path('seo_options/',seo_options,name='seo_options'),
    path('results/',results,name='results'),
    path('exraction/',exraction,name='exraction'),
    path('extract_keywordss/',extract_keywordss,name='extract_keywordss'),
    path('extraction_from_video/',extraction_from_video,name='extraction_from_video'),
    # path('extraction_from_url/',extraction_from_url,name='extraction_from_url'),
    path('extraction_from_text/',extraction_from_text,name='extraction_from_text'),   
    path('home/',home,name='home'),
    path("auth/youtube_affiliated/<int:contact_id>/", views.youtube_authenticate_for_affiliated, name="youtube_authenticate"),
    path("auth/youtube/", views.youtube_authenticate, name="youtube_authenticate"),
    path("auth/callback", views.callback, name="callback"),
    path("save_mobile/",views.save_mobile,name="save_mobile"),
    path('onboarding/', views.onboarding, name='onboarding'),
    path('onboarding-action/', views.onboarding_action, name='onboarding_action'),
    path('goal_selection/',views.goal_selection,name='goal_selection'),
    path('discovery_selection/',views.discovery_selection,name='discovery_selection'),
    path('save_discovery_data',views.save_discovery_data,name='save_discovery_data'),
    path('save_goal_data',views.save_goal_data,name='save_goal_data'),
    path('paymentselection',views.paymentselection,name='paymentselection'),
    path('create_order/<int:amount>/', views.create_order, name='create_order'),
    path('payment/success/', views.payment_success, name='payment_success'),
    path('samplepage/',views.samplepage,name='samplepage'),
    path('checkout/',views.checkout,name='checkout'),
    path('affiliate_dashboard/',views.affiliate_dashboard,name='affiliate_dashboard'),
    path('mobilenumber/',views.mobilenumber,name='mobilenumber'),
    path('contactsave/',views.contactsave,name='contactsave'),
    path('save-payment-details/', views.save_payment_details, name='save_payment_details'),
    path('get_related_keywords_trends/',views.get_related_keywords_trends,name='get_related_keywords_trends'),
    path('extraction_from_text_api/',views.extraction_from_text_api,name='extraction_from_text_api'),
    path('extraction_from_url_using_scrapy/',views.extraction_from_url_using_scrapy,name='extraction_from_url_using_scrapy'),
    path('extraction_from_url_api/',views.extraction_from_url_api,name='extraction_from_url_api'),
    path('user_dashboard/',views.user_dashboard,name='user_dashboard'),
    path('youtube_url/',views.youtube_url,name='youtube_url')

]