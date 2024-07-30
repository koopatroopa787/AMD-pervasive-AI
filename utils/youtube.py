from pytube import YouTube

# Ensure correct YouTube client versions
from pytube.innertube import _default_clients

_default_clients["ANDROID"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["ANDROID_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_MUSIC"]["context"]["client"]["clientVersion"] = "6.41"
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

def download_youtube_video(url, output_path):
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=False).first()
    video_path = stream.download(output_path=output_path, filename="video.mp4")
    return video_path

def get_video_info(youtube_url):
    yt = YouTube(youtube_url)
    video_info = {
        "video_id": yt.video_id,
        "title": yt.title,
        "author": yt.author,
        "channel_id": yt.channel_id,
        "description": yt.description,
        "publish_date": yt.publish_date.strftime('%Y-%m-%d') if yt.publish_date else 'Unknown',
        "views": yt.views,
        "length": yt.length,
        "rating": yt.rating,
        "keywords": yt.keywords,
        "thumbnail_url": yt.thumbnail_url,
        "video_url": yt.watch_url,
        "captions": {lang: str(yt.captions[lang]) for lang in yt.captions},
        "streams": [str(stream) for stream in yt.streams]
    }
    return video_info
