# Generated by Django 5.1.2 on 2024-10-30 11:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('extractionapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='keyword',
            name='keyword',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='keyword',
            name='search_volume',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='keyword',
            name='source',
            field=models.CharField(max_length=50),
        ),
    ]
