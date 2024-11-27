from scrapy.crawler import CrawlerProcess
from fozato_scrapy_project.spiders.youtube_tags import YouTubeTagsSpider
from scrapy.utils.project import get_project_settings
from pydispatch import dispatcher
from scrapy import signals

class YouTubeSpiderRunner:
    def __init__(self, video_urls=None):
        """
        Initializes the YouTubeSpiderRunner class with video URLs.
        
        Args:
            video_urls (list, optional): List of YouTube video URLs to scrape.
        """
        self.video_urls = video_urls or []
        self.scraped_data = []

    def collect_data(self, item, response, spider):
        """
        Callback function to collect scraped data from the spider.

        Args:
            item (dict): The scraped item.
            response (scrapy.http.Response): The response from the spider.
            spider (scrapy.Spider): The spider object.
        """
        self.scraped_data.append(item)

    def run_spider_for_single_url(self, video_url):
        """
        Runs the Scrapy spider for a single YouTube video URL.

        Args:
            video_url (str): The URL of the YouTube video to scrape.
        
        Returns:
            list: Scraped data for the single video URL.
        """
        # Initialize the Scrapy process with project settings
        process = CrawlerProcess(get_project_settings())

        # Connect Scrapy's signal for item collection to the callback
        dispatcher.connect(self.collect_data, signal=signals.item_scraped)

        # Start the spider for the given video URL
        process.crawl(YouTubeTagsSpider, video_url=video_url)
        process.start()  # Blocks until the crawling process completes

        return self.scraped_data

    def run_spider_for_multiple_urls(self,video_urls):
        """
        Runs the Scrapy spider for multiple YouTube video URLs.

        Args:
            None
        
        Returns:
            list: Scraped data for all video URLs.
        """
        # Initialize the Scrapy process with project settings
        process = CrawlerProcess(get_project_settings())

        # Connect Scrapy's signal for item collection to the callback
        dispatcher.connect(self.collect_data, signal=signals.item_scraped)

        # Crawl for each video URL
        for url in video_urls:
            process.crawl(YouTubeTagsSpider, video_url=url)

        # Start the Scrapy process (blocking call)
        process.start()

        return self.scraped_data
