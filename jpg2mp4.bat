@echo off
ffmpeg -f image2 -i images/Capture_%%03d.jpg -r 30 -vcodec libx265 Capture.mp4
PAUSE