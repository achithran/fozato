# Generated by Django 5.0.6 on 2024-12-07 06:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('extractionapp', '0028_alter_youtubeuser_amount_alter_youtubeuser_currency_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='AffiliateUser',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('referral_code', models.CharField(max_length=8, unique=True)),
            ],
        ),
    ]