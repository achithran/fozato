import scrapy
class YouTubeTagsSpider(scrapy.Spider):
    name = "youtube_tags"

    custom_settings = {
        "DEPTH_LIMIT": 2,  # Set depth limit specific to this spider
    }

    def __init__(self, video_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [video_url] if video_url else []
        self.related_video_urls = []

    def parse(self, response):
        # Extract keywords and description
        keywords = response.css('meta[name="keywords"]::attr(content)').get()
        description = response.css('meta[name="description"]::attr(content)').get()

        yield {
            "video_url": response.url,
            "keywords": keywords.split(",") if keywords else [],
            "description": description or "",
        }

        # Extract related video URLs
        related_videos = response.css('a.yt-simple-endpoint::attr(href)').getall()
        self.related_video_urls = [
            f"https://www.youtube.com{url}" for url in related_videos if "/watch?" in url
        ]

        # Yield related video URLs for scraping
        for video_url in self.related_video_urls:
            yield scrapy.Request(video_url, callback=self.parse)


# import scrapy

# class YouTubeTagsSpider(scrapy.Spider):
#     name = "youtube_tags"

#     def __init__(self, video_url=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.start_urls = [video_url] if video_url else []

#     def parse(self, response):
#         # Extract metadata from meta tags
#         title = response.css('meta[name="title"]::attr(content)').get()
#         description = response.css('meta[name="description"]::attr(content)').get()
#         keywords = response.css('meta[name="keywords"]::attr(content)').get()

#         yield {
#             "video_url": response.url,
#             "title": title,
#             "description": description,
#             "keywords": keywords.split(",") if keywords else [],
#         }

#         # You can extract related video URLs if needed
#         related_videos = response.css('a.yt-simple-endpoint::attr(href)').getall()
#         for path in related_videos:
#             if "/watch?v=" in path:
#                 yield scrapy.Request(f"https://www.youtube.com{path}", callback=self.parse)





