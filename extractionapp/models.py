from django.db import models
# from django.contrib.auth.models import User

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

    def __str__(self):
        return f"{self.username} ({self.channel_name})"

    
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






