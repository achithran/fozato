# Generated by Django 5.0.6 on 2024-12-06 07:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('extractionapp', '0027_youtubeuser_currency'),
    ]

    operations = [
        migrations.AlterField(
            model_name='youtubeuser',
            name='amount',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=20, null=True),
        ),
        migrations.AlterField(
            model_name='youtubeuser',
            name='currency',
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
        migrations.AlterField(
            model_name='youtubeuser',
            name='payment_plan',
            field=models.CharField(blank=True, choices=[('Basic', 'Basic'), ('Standard', 'Standard'), ('Premium', 'Premium')], max_length=20, null=True),
        ),
        migrations.AlterField(
            model_name='youtubeuser',
            name='payment_term',
            field=models.CharField(blank=True, choices=[('Yearly', 'Yearly'), ('Monthly', 'Monthly')], max_length=20, null=True),
        ),
    ]
