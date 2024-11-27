from scrapy.utils.project import get_project_settings
from django.core.management.base import BaseCommand
from scrapy.crawler import CrawlerProcess
from trend.spiders.gtrend import GtrendSpider  # Correct the import path as needed

class Command(BaseCommand):
    help = 'Runs the Google Trends spider'

    def handle(self, *args, **kwargs):
        try:
            settings = get_project_settings()  # Ensure Scrapy settings are loaded
            process = CrawlerProcess(settings)
            process.crawl(GtrendSpider)  # Correct spider name
            process.start()
            self.stdout.write(self.style.SUCCESS('Spider executed successfully'))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'Error: {e}'))
