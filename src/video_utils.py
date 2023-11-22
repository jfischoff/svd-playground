import ffmpeg
import os

def frames_to_video(input_folder,
                    output_file,
                    pattern='%04d.png',
                    frame_rate=8,
                    vcodec='libx264',
                    crf=18,
                    preset='veryslow',
                    pix_fmt='yuv420p'):
  # Define input file pattern
  input_pattern = os.path.join(input_folder, pattern)

  # Create FFmpeg input stream from image sequence
  input_stream = ffmpeg.input(input_pattern, framerate=frame_rate)

  # create the directory for the output file
  os.makedirs(os.path.dirname(output_file), exist_ok=True)

  output_stream = ffmpeg.output(input_stream,
                                output_file,
                                vcodec=vcodec,
                                crf=crf,
                                preset=preset,
                                pix_fmt=pix_fmt,
                                y='-y')
  # Run FFmpeg command to convert image sequence to video
  ffmpeg.run(output_stream)

if __name__ == '__main__':
  # Define input and output folders
  input_folder = 'outputs/simple_video_sample/svd/00000000'
  output_file = 'outputs/videos/video.mp4'

  # Convert frames to video
  frames_to_video(input_folder, output_file)