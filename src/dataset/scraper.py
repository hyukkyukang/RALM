import concurrent.futures
import re

import hkkang_utils.time as time_utils
import newspaper
import tldextract

# https://stackoverflow.com/questions/7160737/python-how-to-validate-a-url-in-python-malformed-or-not
url_regex = re.compile(
    r"^(?:http)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)

# domains that aren't scraper friendly. do not include subdomains!
exclude_domains = set(
    [
        # image & video hosting sites
        "imgur.com",
        "redd.it",
        "instagram.com",
        "discord.gg",
        "gfycat.com",
        "giphy.com",
        "reddituploads.com",
        "redditmedia.com",
        "twimg.com",
        "sli.mg",
        "magaimg.net",
        "flickr.com",
        "imgflip.com",
        "youtube.com",
        "youtu.be",
        "youtubedoubler.com",
        "vimeo.com",
        "twitch.tv",
        "streamable.com",
        "bandcamp.com",
        "soundcloud.com",
        # not scraper friendly
        "reddit.com",
        "gyazo.com",
        "github.com",
        "xkcd.com",
        "twitter.com",
        "spotify.com",
        "itunes.apple.com",
        "facebook.com",
        "gunprime.com",
        "strawpoll.me",
        "voyagefusion.com",
        "rollingstone.com",
        "google.com",
        "timeanddate.com",
        "walmart.com",
        "roanoke.com",
        "spotrac.com",
        # original paper excluded wikipedia
        "wikipedia.org",
        # lots of top posts for this one
        "battleforthenet.com",
    ]
)

exclude_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".gifv",
    ".pdf",
    ".mp4",
    ".mp3",
    ".ogv",
    ".webm",
    ".doc",
    ".docx",
    ".log",
    ".csv",
    ".dat",
    ".iso",
    ".bin",
    ".exe",
    ".apk",
    ".jar",
    ".app",
    ".ppt",
    ".pps",
    ".pptx",
    ".xml",
    ".gz",
    ".xz",
    ".bz2",
    ".tgz",
    ".tar",
    ".zip",
    ".wma",
    ".mov",
    ".wmv",
    ".3gp",
    ".svg",
    ".rar",
    ".wav",
    ".avi",
    ".7z",
)


class Scraper:
    def __init__(self, timeout: int):
        self.timer: time_utils.Timer = time_utils.Timer()
        self.timeout = timeout

    def should_exclude(
        self, url: str, ext_url: tldextract.tldextract.ExtractResult = None
    ):
        # Define your exclusion logic here
        return False

    def _download_article(self, article):
        article.download()

    def __call__(
        self,
        url: str,
        memoize: bool = False,
        ext_url: tldextract.tldextract.ExtractResult = None,
    ):
        text = ""
        word_count = 0
        elapsed_time = 0
        success = False

        if not self.should_exclude(url, ext_url=ext_url):
            with self.timer.measure():
                try:
                    article = newspaper.Article(
                        url, fetch_images=False, memoize_articles=memoize
                    )

                    # Run article.download() with a timeout
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._download_article, article)
                        try:
                            future.result(timeout=self.timeout)
                            article.parse()
                            text = article.text
                            word_count = len(text.split())
                            success = True
                        except concurrent.futures.TimeoutError:
                            raise TimeoutError(
                                f"Timeout: Article download took more than {self.timeout} seconds."
                            )
                        except Exception as e:
                            raise Exception(f"Exception during article processing: {e}")
                except Exception as e:
                    pass

            elapsed_time = int(self.timer.elapsed_time)

        # Create metadata
        metadata = {
            "url": url,
            "word_count": word_count,
            "elapsed": elapsed_time,
            "success": success,
        }

        # Return the text and metadata
        return text, metadata

    @classmethod
    def should_exclude(
        self, url: str, ext_url: tldextract.tldextract.ExtractResult = None
    ) -> bool:
        if not ext_url:
            ext_url = tldextract.extract(url)
        domain_name = (
            f"{ext_url.domain}.{ext_url.suffix}"  # Full domain (e.g., example.com)
        )
        basedomain = ext_url.suffix  # TLD (e.g., com, org)

        # Ignore non-URLs
        if len(url) <= 8 or " " in url or re.match(url_regex, url) is None:
            return True

        # Ignore excluded domains
        if basedomain in exclude_domains or domain_name in exclude_domains:
            return True

        # Ignore case-insensitive matches for excluded extensions
        if url.lower().split("?")[0].endswith(exclude_extensions):
            return True

        return False
