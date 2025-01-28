# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# WARNING! This file contains a blacklist of known malicious sites and thus contains some NSFW language.

import argparse
import glob
import logging
import re
import time

import tldextract
import tqdm

logger = logging.getLogger("Blacklist_urls")

domain_blacklist = set(
    [
        "500px",
        "aapks",
        "akamaihd",
        "amazon",
        "apple",
        "artifactfire",
        "artstation",
        "awwni",
        "bandcamp",
        "battleforthenet",
        "coinscalendar",
        "dailymotion",
        "deviantart",
        "discord",
        "discordapp",
        "dlapkandroid",
        "dropbox",
        "e621",
        "ebay",
        "edealinfo",
        "erome",
        "eroshare",
        "explosm",
        "facebook",
        "fbcdn",
        "flickr",
        "furaffinity",
        "futhead",
        "gatopardo",
        "gfycat",
        "gifsound",
        "gifsoup",
        "giphy",
        "github",
        "google",
        "gunprime",
        "gyazo",
        "horsefucker",
        "hotdealstar",
        "imagefap",
        "imageshack",
        "imgflip",
        "imgur",
        "instagram",
        "karmadecay",
        "kryptocal",
        "kym-cdn",
        "liveleak",
        "livememe",
        "lmgtfy",
        "magaimg",
        "memegenerator",
        "minorplanetcenter",
        "minus",
        "mobafire",
        "morejpeg",
        "nocookie",
        "pcpartpicker",
        "photobucket",
        "pinimg",
        "pinterest",
        "pixiv",
        "pornhub",
        "prntscr",
        "puu",
        "qkme",
        "quickmeme",
        "radd",
        "redd",
        "reddit",
        "reddit-stream",
        "redditlog",
        "redditmedia",
        "reddituploads",
        "redtube",
        "reupp",
        "reverb",
        "roanoke",
        "rollingstone",
        "sli",
        "soundcloud",
        "soundgasm",
        "spankbang",
        "spotify",
        "strawpoll",
        "streamable",
        "timeanddate",
        "tinypic",
        "touhouradio",
        "tumblr",
        "twimg",
        "twitch",
        "twitter",
        "vid",
        "vimeo",
        "vine",
        "vkaao",
        "vocaroo",
        "voyagefusion",
        "walmart",
        "wciu",
        "wikimedia",
        "wikipedia",
        "xhamster",
        "xkcd",
        "xvideos",
        "youtu",
        "youtube",
        "youtubedoubler",
        "ytimg",
        "zillexplorer",
    ]
)
# List of extentions to blacklist.
extentions_blacklist = (
    ".3gp",
    ".7z" ".ai",
    ".aif",
    ".apk",
    ".app",
    ".avi",
    ".bin",
    ".bmp",
    ".bz2",
    ".css",
    ".csv",
    ".dat",
    ".deb",
    ".dmg",
    ".doc",
    ".docx",
    ".exe",
    ".gif",
    ".gifv",
    ".gz",
    ".iso",
    ".jar",
    ".jpeg",
    ".jpg",
    ".js",
    ".log",
    ".mid",
    ".midi",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".ogv",
    ".otf",
    ".pdf",
    ".pkg",
    ".png",
    ".pps",
    ".ppt",
    ".pptx",
    ".psd",
    ".py",
    ".qt",
    ".ram",
    ".rar",
    ".sql",
    ".svg",
    ".swf",
    ".tar.gz",
    ".tar",
    ".tgz",
    ".tiff",
    ".ttf",
    ".txt",
    ".wav",
    ".webm",
    ".wma",
    ".wmv",
    ".xls",
    ".xlsx",
    ".xml",
    ".xz",
    ".zip",
)


url_regex = re.compile(
    r"^(?:http)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def extention_is_in_blacklist(url):
    if url.split("?")[0].lower().endswith(extentions_blacklist):
        return True
    return False


def domain_is_in_blacklist(url):
    domain = tldextract.extract(url).domain
    return domain in domain_blacklist


def print_blacklist(string: str, do_print: bool = False):
    if do_print:
        logger.info(string)


def url_is_malformed(url):
    return re.match(url_regex, url) is None


def print_progress(
    prefix,
    start_time,
    urls_counter,
    domain_blacklist_counter,
    extention_blacklist_counter,
    short_url_counter,
    malformed_url_counter,
    duplicate_url_counter,
):
    string = prefix + " | "
    string += "time elapsed (s): {:.2f} | ".format(time.time() - start_time)
    string += "number of urls: {} | ".format(urls_counter)
    string += "domain blacklisted: {} | ".format(domain_blacklist_counter)
    string += "extention blacklisted: {} | ".format(extention_blacklist_counter)
    string += "short urls (<=8): {} | ".format(short_url_counter)
    string += "malformed urls: {} | ".format(malformed_url_counter)
    string += "duplicate urls: {}".format(duplicate_url_counter)
    logger.info(string)


def main(
    url_dir_path: str,
    output_file_path: str,
    do_print_blacklist: bool = False,
) -> None:
    # List of the domains to blacklist.

    # Malformed urls.
    # This function is adapted from:
    #   https://stackoverflow.com/questions/7160737/python-how-to-validate-a-url-in-python-malformed-or-not

    logger.info("remove blacklisted urls ..")

    # Get the list of url files.
    files = glob.glob(url_dir_path + "/*.txt")
    logger.info("> found {} files".format(len(files)))

    urls = set()
    urls_counter = 0
    domain_blacklist_counter = 0
    extention_blacklist_counter = 0
    short_url_counter = 0
    malformed_url_counter = 0
    duplicate_url_counter = 0
    start_time = time.time()
    for filename in tqdm.tqdm(files, desc="Processing files"):
        with open(filename, "r") as f:
            for line in tqdm.tqdm(f, desc="Processing urls"):
                url = line.strip()
                urls_counter += 1
                if domain_is_in_blacklist(url):
                    print_blacklist(
                        f"[DOMAIN BLACKLIST]: {url}", do_print=do_print_blacklist
                    )
                    domain_blacklist_counter += 1
                elif extention_is_in_blacklist(url):
                    print_blacklist(
                        f"[EXTENTION BLACKLIST]: {url}",
                        do_print=do_print_blacklist,
                    )
                    extention_blacklist_counter += 1
                elif len(url) <= 8:
                    print_blacklist(f"[SHORT URL]: {url}", do_print=do_print_blacklist)
                    short_url_counter += 1
                elif url_is_malformed(url):
                    print_blacklist(
                        f"[MALFORMED URL]: {url}", do_print=do_print_blacklist
                    )
                    malformed_url_counter += 1
                elif url in urls:
                    print_blacklist(
                        f"[DUPLICATE URL]: {url}", do_print=do_print_blacklist
                    )
                    duplicate_url_counter += 1
                else:
                    urls.add(url)
                if urls_counter % 100000 == 0:
                    print_progress(
                        "PROGRESS",
                        start_time,
                        urls_counter,
                        domain_blacklist_counter,
                        extention_blacklist_counter,
                        short_url_counter,
                        malformed_url_counter,
                        duplicate_url_counter,
                    )

    print_progress(
        "FINAL",
        start_time,
        urls_counter,
        domain_blacklist_counter,
        extention_blacklist_counter,
        short_url_counter,
        malformed_url_counter,
        duplicate_url_counter,
    )

    # Write the final set of urls.
    logger.info("> writing cleaned up url list to {}".format(output_file_path))
    with open(output_file_path, "w") as f:
        for url in urls:
            f.write(url + "\n")

    logger.info("Done !")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print_blacklist", action="store_true")
    parser.add_argument("--url_dir_path", type=str, default="")
    parser.add_argument("--output_file_path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_arguments

    main(
        url_dir_path=args.url_dir_path,
        output_file_path=args.output_file_path,
        do_print_blacklist=args.print_blacklist,
    )
