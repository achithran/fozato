# Generated by Django 5.0.6 on 2024-12-09 07:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('extractionapp', '0037_paymentplan'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='payment',
            name='payment_term',
        ),
        migrations.AddField(
            model_name='payment',
            name='razorpay_order_id',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='payment',
            name='razorpay_signature',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]