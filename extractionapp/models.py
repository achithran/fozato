from django.db import models
from django.contrib.auth.models import User

class Keyword(models.Model):
    keyword = models.CharField(max_length=255)
    search_volume = models.IntegerField(null=True, blank=True)  # Allow null values
    source = models.CharField(max_length=50)  # e.g., Google, YouTube

    def __str__(self):
        return self.keyword
class videoSEODB(models.Model):
    video_file=models.FileField(upload_to='media/videos/')
    base_file_name = models.CharField(max_length=255, default='unknown_video.mp4') 
    title=models.CharField(max_length=255)
    description=models.TextField()
    tags=models.TextField()
    keywords=models.TextField()

    def __str__(self):
        return self.title
    
class urlSEODB(models.Model):
    video_url=models.TextField()
    title=models.TextField()
    description=models.TextField()
    tags=models.TextField()
    keywords=models.TextField()

    def __str__(self):
        return self.title
    
class keywordSEODB(models.Model):
    key=models.TextField()
    title=models.CharField(max_length=255)
    description=models.TextField()
    tags=models.TextField()
    keywords=models.TextField()    

    def __str__(self):
        return self.title

class YouTubeUser(models.Model):
    username = models.CharField(max_length=255)
    channel_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    mobile_number = models.CharField(max_length=20, blank=True, null=True) 
    role = models.CharField(max_length=50,null=True, blank=True)
    goal = models.CharField(max_length=100,null=True, blank=True) 
    discovery=models.CharField(max_length=100,null=True, blank=True)
    free_trial_start_date = models.DateTimeField(null=True, blank=True)
    trial_status = models.CharField(max_length=20, default='Active', choices=[('Active', 'Active'), ('Expired', 'Expired')])
    payment_term = models.CharField(
        max_length=20,
        choices=[('Yearly', 'Yearly'), ('Monthly', 'Monthly')],
        null=True,
        blank=True
    )
    payment_plan = models.CharField(
        max_length=20,
        choices=[('Basic', 'Basic'), ('Standard', 'Standard'), ('Premium', 'Premium')],
        null=True,
        blank=True
    )
    amount = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    currency = models.CharField(max_length=10, null=True, blank=True)  # Field to store the currency (USD, INR)



    def __str__(self):
        return f"{self.username} ({self.channel_name})"

class Subscription_Data(models.Model):
    youtube_user = models.ForeignKey('YouTubeUser', on_delete=models.CASCADE)  # Link to YouTubeUser
    plan_name = models.CharField(max_length=100)  # e.g., 'Basic', 'Premium', etc.
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    razorpay_subscription_id = models.CharField(max_length=255, unique=True,null=True)
    status = models.CharField(max_length=20, choices=[('active', 'Active'), ('cancelled', 'Cancelled')], default='active',null=True)


    def __str__(self):
        return f"{self.plan_name} subscription for {self.youtube_user.username}"
    

class Payment(models.Model):
    youtube_user = models.ForeignKey('YouTubeUser', on_delete=models.CASCADE)  # Link to YouTubeUser
    payment_id = models.CharField(max_length=255)  # Razorpay payment ID
    subscription = models.ForeignKey('Subscription_Data', on_delete=models.SET_NULL, null=True, blank=True)  # Link to subscription
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=10, default='INR')  # Currency like 'INR', 'USD'
    status = models.CharField(max_length=20, choices=[('success', 'Success'), ('failed', 'Failed'), ('pending', 'Pending')])
    payment_date = models.DateTimeField()
    razorpay_order_id = models.CharField(max_length=255, null=True, blank=True)  # Razorpay order ID
    razorpay_signature = models.CharField(max_length=255, null=True, blank=True)  # Razorpay signature
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Payment for {self.youtube_user.username} - {self.amount} {self.currency}"

class PaymentPlan(models.Model):
    CURRENCY_CHOICES = [
        ('INR', 'Indian Rupee'),
        ('USD', 'US Dollar'),
    ]
    
    name = models.CharField(max_length=100)  # e.g., 'Basic', 'Premium'
    plan_id = models.CharField(max_length=255, unique=True)  # Razorpay Plan ID
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    interval = models.CharField(max_length=10)  # e.g., 'monthly', 'yearly'
    currency = models.CharField(max_length=10, choices=CURRENCY_CHOICES, default='INR')  # Currency with choices
    
    def __str__(self):
        return f"{self.name} ({self.currency})"


    
class UserRole(models.Model):
    username = models.CharField(max_length=255)
    email = models.EmailField()
    role = models.CharField(max_length=50)

    def __str__(self):
        return self.username
    
class UserGoal(models.Model):
    # user = models.ForeignKey(User, on_delete=models.CASCADE)
    channel_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    role = models.CharField(max_length=50)
    goal = models.CharField(max_length=100)    

class Discovery(models.Model):
    channel_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    role = models.CharField(max_length=50)
    goal = models.CharField(max_length=100)
    discovery=models.CharField(max_length=100) 

class Subscription(models.Model):
    channel_name = models.CharField(max_length=255)
    email = models.EmailField()
    role = models.CharField(max_length=100)
    goal = models.TextField()
    discovery = models.TextField()
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    plan_duration = models.CharField(max_length=10, choices=[('monthly', 'Monthly'), ('yearly', 'Yearly')])
    payment_id = models.CharField(max_length=100, null=True, blank=True)
    order_id = models.CharField(max_length=100, null=True, blank=True)
    signature = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.email} - {self.plan_duration}"  

class ContactForm(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=15)
    country_code = models.CharField(max_length=5)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    authentication_flag = models.BooleanField(default=False)
    
    

    def __str__(self):
        return f"{self.first_name} {self.last_name}"    

class PaymentDetails(models.Model):
    payment_id = models.CharField(max_length=100, null=True, blank=True)
    order_id = models.CharField(max_length=100, null=True, blank=True)
    signature = models.TextField(null=True, blank=True)


class AffiliateUser(models.Model):
    email = models.EmailField(unique=True)
    referral_code = models.CharField(max_length=20, unique=True)
    user_logged_in_date = models.DateTimeField(null=True, blank=True)  # Store the login date
    username = models.CharField(max_length=255,null=True, blank=True)
    channel_name = models.CharField(max_length=255,null=True, blank=True)
    free_trial_start_date = models.DateTimeField(null=True, blank=True)
    trial_status = models.CharField(max_length=20, default='Active', choices=[('Active', 'Active'), ('Expired', 'Expired')])
    user_email = models.EmailField(unique=True,null=True, blank=True)


    def __str__(self):
        return self.email





