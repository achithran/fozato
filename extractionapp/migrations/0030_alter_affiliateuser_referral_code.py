# Generated by Django 5.0.6 on 2024-12-07 07:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('extractionapp', '0029_affiliateuser'),
    ]

    operations = [
        migrations.AlterField(
            model_name='affiliateuser',
            name='referral_code',
            field=models.CharField(max_length=20, unique=True),
        ),
    ]
