# Generated by Django 5.1.2 on 2024-11-15 08:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('extractionapp', '0015_subscription'),
    ]

    operations = [
        migrations.AddField(
            model_name='subscription',
            name='order_id',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='subscription',
            name='payment_id',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='subscription',
            name='signature',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]