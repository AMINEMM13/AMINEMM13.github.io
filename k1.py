import os
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import json
import time
try:
    from pydub import AudioSegment
except ImportError:
    class AudioSegment:
        @staticmethod
        def from_file(file, format=None):
            return AudioSegment()
        def export(self, out_f, format=None):
            pass
import textwrap
try:
    from moviepy import ImageClip, TextClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip, VideoClip
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError:
    # Pyodide 환경에서 moviepy 미설치 시 임시 더미 정의
    class ImageClip: pass
    class TextClip: pass
    class AudioFileClip: pass
    class CompositeVideoClip: pass
    def concatenate_videoclips(clips): return clips[0] if clips else None
    class VideoFileClip: pass
    class VideoClip: pass
    class ImageSequenceClip: pass
import asyncio
import re
import edge_tts
import pyworld as pw
import soundfile as sf
import shutil  # 결과 폴더로 파일 복사에 사용
import sys

# 진행률 파일 기록 함수 추가 (추가된 부분)
BASE = os.path.dirname(__file__)
def write_progress(pct: int):
    """outputs/progress.txt 에 'PROGRESS:xx' 형식으로 진행률 기록"""
    prog_dir = os.path.join(BASE, "outputs")
    os.makedirs(prog_dir, exist_ok=True)
    progfile = os.path.join(prog_dir, "progress.txt")
    mode = "w" if pct == 0 else "a"
    with open(progfile, mode, encoding="utf-8") as f:
        f.write(f"PROGRESS:{pct}\n")
# ────────────────────────────────────────────────────────────────────────────

# ZoomImageSequenceClip 클래스
class ZoomImageSequenceClip(ImageSequenceClip):
    def __init__(
        self,
        sequence,
        fps=None,
        durations=None,
        with_mask=True,
        is_mask=False,
        load_images=False,
        start_scale=1.0,
        end_scale=2.0
    ):
        super().__init__(
            sequence=sequence,
            fps=fps,
            durations=durations,
            with_mask=with_mask,
            is_mask=is_mask,
            load_images=load_images
        )
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.total_duration = self.duration

        old_frame_function = self.frame_function

        def zoom_frame_function(t):
            scale = self.start_scale + (self.end_scale - self.start_scale) * (t / self.total_duration)
            frame = old_frame_function(t)
            h_frame, w_frame, _ = frame.shape
            new_w = int(w_frame * scale)
            new_h = int(h_frame * scale)
            pil_img = Image.fromarray(frame)
            resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if new_w > w_frame or new_h > h_frame:
                left = (new_w - w_frame) // 2
                top = (new_h - h_frame) // 2
                right = left + w_frame
                bottom = top + h_frame
                cropped = resized.crop((left, top, right, bottom))
                return np.array(cropped)
            else:
                return np.array(resized)

        self.frame_function = zoom_frame_function
        self.size = self.frame_function(0).shape[:2][::-1]

# lower_pitch 함수
def lower_pitch(input_file, output_file, pitch_factor=1.0, formant_factor=1.0):
    y, sr = sf.read(input_file)
    y = y.astype(np.float64)
    if y.ndim > 1:
        y = y[:, 0]
    f0, sp, ap = pw.wav2world(y, sr)
    f0_adjusted = f0 * pitch_factor
    num_bins = sp.shape[1]
    sp_adjusted = np.empty_like(sp)
    for i in range(sp.shape[0]):
        old_bins = np.arange(num_bins)
        new_bins = np.clip(old_bins * formant_factor, 0, num_bins - 1)
        sp_adjusted[i, :] = np.interp(old_bins, new_bins, sp[i, :])
    y_adjusted = pw.synthesize(f0_adjusted, sp_adjusted, ap, sr)
    sf.write(output_file, y_adjusted, sr, format='MP3')

# [DEBUG] 로깅 함수
def debug_log(message):
    print(f"[DEBUG]: {message}")

# 헤드라인 텍스트 이미지 생성
def create_headline_text_image(text, font_path, target_resolution):
    img = Image.new("RGBA", (target_resolution[0], target_resolution[1]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 200)
    virtual_box_width = 1080
    line_height = 250
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] <= virtual_box_width:
            current_line = test_line
        else:
            if current_line.strip():
                lines.append(current_line.strip())
            current_line = word + " "
    if current_line.strip():
        lines.append(current_line.strip())
    total_text_height = line_height * len(lines)
    start_y = (target_resolution[1] - total_text_height) // 2
    y = start_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        centered_x = (target_resolution[0] - text_width) // 2
        draw.text((centered_x, y), line, font=font, fill=(255, 255, 0))
        y += line_height
    return img

# 비디오 클립 커스터마이징 함수, 필름 카운트다운, TTS 생성 등 이하 원본 로직 그대로 유지…

async def main():
    try:
        write_progress(0)
        if len(sys.argv) != 4:
            raise ValueError("사용법: python k1.py <input_number> <selected_folder> <apply_laugh>")
        
        input_number = sys.argv[1]
        selected_folder = sys.argv[2]
        apply_laugh = sys.argv[3]

        if not input_number.isdigit():
            raise ValueError("숫자를 입력해야 합니다.")
        if not selected_folder:
            raise ValueError("폴더를 선택해야 합니다.")
        if apply_laugh not in ["1", "2"]:
            raise ValueError("웃음소리 옵션은 1 또는 2이어야 합니다.")
        
        use_laugh = apply_laugh == "1"

        base_dir = os.path.join(BASE, "uploads", "iv_shorts")
        media_dir = os.path.join(base_dir, "iv", selected_folder)
        text_file = os.path.join(base_dir, f"{input_number}.txt")
        font_path = os.path.join(BASE, "static", "fonts", "sbugrob.ttf")
        clock_audio_path = os.path.join(BASE, "static", "audio", "clock", "clock.mp3")
        laugh_audio_path = os.path.join(BASE, "static", "audio", "clock", "laugh.mp3")
        result_base = os.path.join(BASE, "outputs", "result")
        
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"텍스트 파일이 없습니다: {text_file}")
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"폰트 파일이 없습니다: {font_path}")
        if not os.path.exists(clock_audio_path):
            raise FileNotFoundError(f"클럭 오디오 파일이 없습니다: {clock_audio_path}")
        if use_laugh and not os.path.exists(laugh_audio_path):
            raise FileNotFoundError(f"웃음 오디오 파일이 없습니다: {laugh_audio_path}")

        with open(text_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError("텍스트 파일이 비어 있습니다")

        headline = lines[0]
        debug_log(f"헤드라인: {headline}")
        write_progress(10)

        media_files = [f for f in os.listdir(media_dir) if os.path.isfile(os.path.join(media_dir, f))]
        def get_sort_key(filename):
            base = os.path.splitext(filename)[0]
            m = re.match(r'^(\d+)([a-zA-Z]*)$', base)
            if m:
                number_part = int(m.group(1))
                letter_part = m.group(2)
                return (number_part, letter_part)
            else:
                return (float('inf'), base)
        media_files = sorted(media_files, key=get_sort_key)
        media_paths = [os.path.join(media_dir, f) for f in media_files]
        if not media_paths:
            raise ValueError("미디어 파일이 없습니다")
        
        debug_log(f"미디어: {media_paths}")

        target_resolution = (1080, 1920)
        clips = []
        temp_files = []
        bg_offset = 0

        first_media = media_paths[0]
        ext = os.path.splitext(first_media)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            img = Image.open(first_media).resize(target_resolution, Image.Resampling.LANCZOS)
            base_clip = ImageClip(np.array(img)).with_duration(0.1)
        else:
            bg_clip = VideoFileClip(first_media)
            base_clip = get_video_segment(bg_clip, 0, 0.1, target_resolution)

        headline_img = create_headline_text_image(headline, font_path, target_resolution)
        headline_clip = ImageClip(np.array(headline_img)).with_duration(0.1).with_position("center")
        clips.append(CompositeVideoClip([base_clip, headline_clip], size=target_resolution))

        image_effect_counter = 0

        for idx, line in enumerate(lines[1:], 1):
            has_slash = line.endswith('/')
            if has_slash:
                line = line[:-1].strip()

            m = re.search(r'^(.*?)(\s*[0-1]|\s*)$', line)
            if m:
                text = m.group(1).strip()
                speaker_num = m.group(2).strip() if m.group(2).strip() in ["0", "1"] else None
            else:
                text = line
                speaker_num = None

            speaker_code = speaker_num if speaker_num in ["0", "1"] else "2"
            debug_log(f"행 {idx}: {text} - 화자 코드: {speaker_code}")

            audio = os.path.join(base_dir, f"audio_{idx}.mp3")
            temp_files.append(audio)
            await generate_tts(text, speaker_code, audio)
            audio_clip = AudioFileClip(audio)
            duration = audio_clip.duration

            media_idx = idx % len(media_paths) if media_paths else 0
            media = media_paths[media_idx]
            debug_log(f"행 {idx}: {text} - {media} (화자: {speaker_code})")

            ext = os.path.splitext(media)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                if image_effect_counter % 2 == 0:
                    media_clip = create_zoom_clip(media, duration, start_scale=1.0, end_scale=2.0)
                else:
                    img_clip = ImageClip(media).with_duration(duration)
                    media_clip = panning_effect(img_clip, duration, direction="left")
                image_effect_counter += 1
            else:
                bg_clip = VideoFileClip(media)
                media_clip = get_video_segment(bg_clip, bg_offset, duration, target_resolution)
                bg_offset += duration

            wrapped = textwrap.wrap(text, width=13)
            text_clips = [
                TextClip(
                    text=part,
                    font=font_path,
                    font_size=70,
                    color="white",
                    method="label",
                    size=(target_resolution[0] - 100, None)
                ).with_duration(duration / len(wrapped))
                for part in wrapped
            ]
            subtitle = concatenate_videoclips(text_clips) if text_clips else TextClip("", font_size=1, color="white").with_duration(duration)
            normal_clip = CompositeVideoClip([media_clip, subtitle.with_position("center")], size=target_resolution).with_duration(duration).with_audio(audio_clip)
            clips.append(normal_clip)

            if has_slash:
                if ext in ['.jpg', '.jpeg', '.png']:
                    freeze_clip = ImageClip(media).with_duration(5)
                else:
                    bg_clip = VideoFileClip(media)
                    freeze_clip = get_video_segment(bg_clip, bg_offset, 5, target_resolution)
                    bg_offset += 5

                countdown_clip = VideoClip(lambda t: create_film_countdown_frame(t, total_duration=5, start_number=5, width=1080, height=1920, circle_diameter=600, circle_color=(128, 100, 200), font_path=font_path), duration=5)
                c_mask = countdown_clip.to_mask()
                countdown_clip = countdown_clip.with_mask(c_mask)
                combined_clip = CompositeVideoClip([freeze_clip.with_duration(5), countdown_clip.with_duration(5)], size=target_resolution).with_duration(5)
                clock_audio_clip = AudioFileClip(clock_audio_path).with_duration(5)
                combined_clip = combined_clip.with_audio(clock_audio_clip)
                clips.append(combined_clip)

            if idx == len(lines[1:]) and use_laugh:
                laugh_audio = AudioFileClip(laugh_audio_path).with_duration(3)
                if ext in ['.jpg', '.jpeg', '.png']:
                    laugh_clip = ImageClip(media).with_duration(3)
                else:
                    bg_clip = VideoFileClip(media)
                    laugh_clip = get_video_segment(bg_clip, bg_offset, 3, target_resolution)
                final_laugh_clip = CompositeVideoClip([laugh_clip], size=target_resolution).with_audio(laugh_audio)
                clips.append(final_laugh_clip)

        write_progress(50)
        final = concatenate_videoclips(clips)
        if hasattr(final, 'mask'):
            final.mask = None

        safe_headline = ''.join(c for c in headline if c not in r'[\\/:*?"<>|]')
        output_name = f"{input_number}{safe_headline}"
        result_folder = os.path.join(result_base, output_name)
        os.makedirs(result_folder, exist_ok=True)
        output = os.path.join(result_folder, f"{output_name}.mp4")

        shocking_text = "shocking1"
        video_duration = final.duration
        half_duration = video_duration / 2

        logo_clip1 = TextClip(text=shocking_text, font=font_path, font_size=40, color="white", method="label").with_opacity(0.2).with_position(("center", 1100)).with_duration(half_duration)
        logo_clip2 = TextClip(text=shocking_text, font=font_path, font_size=40, color="white", method="label").with_opacity(0.2).with_position(("right", "top")).with_duration(video_duration - half_duration).with_start(half_duration)

        final_with_logo = CompositeVideoClip([final, logo_clip1, logo_clip2], size=target_resolution, bg_color=None)
        write_progress(80)
        final_with_logo.write_videofile(output, fps=24, codec="libx264", audio_codec="aac", preset="medium")
        print("비디오 생성 완료!")
        write_progress(100)
        shutil.copy(text_file, os.path.join(result_folder, f"{output_name}.txt"))
        for f in media_files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                src = os.path.join(media_dir, f)
                dst = os.path.join(result_folder, f)
                shutil.copy(src, dst)

        for temp in temp_files:
            if os.path.exists(temp):
                os.remove(temp)

    except Exception as e:
        debug_log(f"에러: {e}")
        traceback.print_exc()
        write_progress(100)

if __name__ == "__main__":
    asyncio.run(main())
