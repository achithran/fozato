# Generated by Django 5.1.2 on 2024-11-18 05:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('extractionapp', '0019_paymentdetails_remove_contactform_order_id_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='contactform',
            name='authentication_flag',
            field=models.BooleanField(default=False),
        ),
    ]
