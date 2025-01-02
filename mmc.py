#!/usr/bin/env python
import argparse
import sys
import os
import shutil
import tempfile
import subprocess
import traceback
from functools import lru_cache
from dataclasses import dataclass
import typing

import filetype


VIDEO_DEFAULT_CRF = 28

@dataclass
class Quality:
    img_target_size: int
    video_target_size: int

DEFAUT_QUALITY = "good"

QUALITIES = {
    "low": Quality(
        img_target_size=640,
        video_target_size=360,
    ),
    "medium": Quality(
        img_target_size=1024,  # 10cm, 240dpi
        video_target_size=480,
    ),
    "good": Quality(
        img_target_size=1440,
        video_target_size=720,
    ),
}


class BadArgumentError(Exception):
    pass


class MassMediaCompressor:
    def __init__(
        self,
        copy_unhandled=True,
        copy_failed=True,
        quality=DEFAUT_QUALITY,
        img_target_size: int | None = None,
        video_target_size: int | None = None,
        video_target_crf: int | None = None,
        log_file: str | typing.IO | None = None,
    ):
        self.copy_unhandled = copy_unhandled
        self.copy_failed = copy_failed
        self.quality = QUALITIES[quality]
        if img_target_size is not None:
            self.quality.img_target_size = img_target_size
        if video_target_size is not None:
            self.quality.video_target_size = video_target_size
        self.video_target_crf = video_target_crf or VIDEO_DEFAULT_CRF
        if log_file is None:
            log_file = "mmc.log"
        elif log_file == "stdout":
            log_file = sys.stdout
        elif log_file == "stderr":
            log_file = sys.stderr
        self.log_file = open(log_file, 'a') if type(log_file) == str else log_file

    def run(self, input: str, output: str | None = None, overwrite=False):
        if not os.path.exists(input):
            raise BadArgumentError("input does not exist")
        if (output is None) == (overwrite is False):
            raise BadArgumentError("output must be defined or overwrite should be true")
        if output is not None:
            output_is_file = "." in os.path.basename(output)
            if not output_is_file and os.path.isfile(input):
                output = os.path.join(output, os.path.basename(input))
            if output_is_file and os.path.isdir(input):
                raise BadArgumentError("if input is a directory, output must also be a directory")
        else:
            output = input
        self.compress_any(input, output)

    def compress_any(self, ifpath: str, ofpath: str):
        if os.path.isdir(ifpath):
            self.compress_dir(ifpath, ofpath)
        elif os.path.isfile(ifpath):
            self.log_info(f"Processing {ifpath}... ")
            try:
                res_status = self.compress_file(ifpath, ofpath)
            except Exception as err:
                self.log_info("[ERROR]")
                if self.copy_failed and copy_file(ifpath, ofpath):
                    self.log_info("[COPIED]\n")
                else:
                    self.log_info("[SKIPPED]\n")
                traceback.print_exc()
            else:
                self.log_info(f"[{res_status}]\n")

    def compress_dir(self, ifpath: str, ofpath: str):
        for fname in os.listdir(ifpath):
            self.compress_any(
                os.path.join(ifpath, fname),
                os.path.join(ofpath, fname),
            )

    def compress_file(self, ifpath: str, ofpath: str):
        if ifpath != ofpath and os.path.isfile(ofpath):
            return "SKIPPED"
        ftype = filetype.guess(ifpath)
        if ftype is None:
            if self.copy_unhandled and copy_file(ifpath, ofpath):
                return "COPIED"
            return "SKIPPED"
        mime = ftype.mime
        media = None
        if mime.startswith("image/"):
            media = ImageMedia(mmc, ifpath)
        elif mime.startswith("audio/"):
            media = AudioMedia(mmc, ifpath)
        elif mime.startswith("video/"):
            media = VideoMedia(mmc, ifpath)
        if media is None:
            if self.copy_unhandled and copy_file(ifpath, ofpath):
                return "COPIED"
            return "SKIPPED"
        if media.is_compressed():
            if copy_file(ifpath, ofpath):
                return "COPIED"
            return "SKIPPED"
        else:
            with tempfile.TemporaryDirectory() as tmpd:
                tmpfpath = os.path.join(tmpd, os.path.basename(ifpath))
                media.compress(tmpfpath)
                shutil.copystat(ifpath, tmpfpath)
                copy_file(tmpfpath, ofpath)
                return "COMPRESSED"
    
    def log_info(self, msg):
        print(msg, end="", flush=True)
    
    def log_error(self, msg):
        print("ERROR", msg, end="", flush=True, file=sys.stderr)


class Media:
    def __init__(self, mmc: MassMediaCompressor, ifpath: str):
        self.mmc = mmc
        self.ifpath = ifpath
    def is_compressed(self):
        return NotImplementedError()
    def compress(self, ofpath: str) -> bool:
        return NotImplementedError()


class ImageMedia(Media):

    def is_compressed(self):
        img_target_size = self.mmc.quality.img_target_size
        width, height = self.get_img_resolution()
        return width <= img_target_size or height <= img_target_size

    def compress(self, ofpath: str):
        img_target_size = self.mmc.quality.img_target_size
        width, height = self.get_img_resolution()
        if width < height and width > img_target_size:
            width = img_target_size
            height = -1
        elif height < width and height > img_target_size:
            height = img_target_size
            width = -1
        else:
            return
        cmd = [
            shutil.which("ffmpeg"),
            "-i", self.ifpath,
            "-filter:v", f"scale={width}:{height}",
            ofpath,
        ]
        res = subprocess.run(
            cmd,
            stdout=self.mmc.log_file,
            stderr=self.mmc.log_file,
        )
        assert res.returncode == 0

    @lru_cache
    def get_img_resolution(self) -> tuple[int, int]:
        cmd = [
            shutil.which("ffprobe"),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            self.ifpath
        ]
        res = subprocess.run(cmd, capture_output=True)
        assert res.returncode == 0
        infos = res.stdout.decode("utf-8").split("x")
        width = int(infos[0])
        height = int(infos[1])
        return width, height



class AudioMedia(Media):
    pass


class VideoMedia(Media):

    def is_compressed(self):
        video_target_size = self.mmc.quality.video_target_size
        width, height = self.get_video_infos()
        return width <= video_target_size or height <= video_target_size

    def compress(self, ofpath: str):
        video_target_size = self.mmc.quality.video_target_size
        width, height = self.get_video_infos()
        if width < height and width > video_target_size:
            width = video_target_size
            height = -2
        elif height < width and height > video_target_size:
            height = video_target_size
            width = -2
        else:
            return
        cmd = [
            shutil.which("ffmpeg"),
            "-i", self.ifpath,
            "-vcodec", "libx265",
            "-crf", str(self.mmc.video_target_crf),
            "-vf", f"scale={width}:{height}",
            ofpath,
        ]
        res = subprocess.run(
            cmd,
            stdout=self.mmc.log_file,
            stderr=self.mmc.log_file,
        )
        assert res.returncode == 0

    @lru_cache
    def get_video_infos(self) -> tuple[int, int, float, int]:
        res = subprocess.run(
            [
                shutil.which("ffprobe"),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "default=noprint_wrappers=1:nokey=1",
                self.ifpath,
            ],
            capture_output=True,
        )
        assert res.returncode == 0
        infos = res.stdout.decode().splitlines()
        width = int(infos[0])
        height = int(infos[1])
        return width, height


def copy_file(ifpath, ofpath):
    if ifpath == ofpath:
        return False
    os.makedirs(os.path.dirname(ofpath), exist_ok=True)
    shutil.copy2(ifpath, ofpath)  # copy2 copies stats
    return True
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--not-copy-unhandled', action='store_true')
    parser.add_argument('--not-copy-failed', action='store_true')
    parser.add_argument('-q', '--quality', default=DEFAUT_QUALITY)
    parser.add_argument('--img-target-size', type=int)
    parser.add_argument('--video-target-size', type=int)
    parser.add_argument('--video-target-crf', type=int)
    parser.add_argument('-o', '--output')
    parser.add_argument('-l', '--log', type=str)
    args = parser.parse_args()
    input_path = args.input
    if not input_path:
        res = input('Write path of (or simply drag and drop) directory to process:\n')
        input_path = res.strip('"')
    try:
        mmc = MassMediaCompressor(
            copy_unhandled=(not args.not_copy_unhandled),
            copy_failed=(not args.not_copy_failed),
            quality=args.quality,
            img_target_size=args.img_target_size,
            video_target_size=args.video_target_size,
            video_target_crf=args.video_target_crf,
            log_file=args.log,
        )
        mmc.run(input_path, args.output, overwrite=args.overwrite)
    except BadArgumentError as err:
        print("ERROR", err, file=sys.stderr)
        sys.exit(1)
