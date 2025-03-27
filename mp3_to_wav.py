#!/usr/bin/env python3
import argparse
import os
from pydub import AudioSegment

def convert_mp3_to_wav(input_file, output_file=None, sample_rate=44100, channels=2, bits=16):
    """
    Convert an MP3 file to WAV format
    
    Args:
        input_file (str): Path to the input MP3 file
        output_file (str, optional): Path to save the output WAV file. If not provided,
                                     will use the same name as input but with .wav extension
        sample_rate (int, optional): Sample rate for the output WAV file. Defaults to 44100 Hz.
        channels (int, optional): Number of audio channels. Defaults to 2 (stereo).
        bits (int, optional): Bit depth. Defaults to 16 bits.
    
    Returns:
        str: Path to the created WAV file
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".wav"
    
    # Load the MP3 file
    audio = AudioSegment.from_mp3(input_file)
    
    # Set audio parameters
    if sample_rate != 44100:
        audio = audio.set_frame_rate(sample_rate)
    if channels != 2:
        audio = audio.set_channels(channels)
    if bits != 16:
        audio = audio.set_sample_width(bits // 8)  # Convert bits to bytes
    
    # Export as WAV
    audio.export(output_file, format="wav")
    
    return output_file

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Convert MP3 audio file to WAV format')
    
    # Add arguments
    parser.add_argument('input_file', help='Path to the input MP3 file')
    parser.add_argument('-o', '--output', help='Path to save the output WAV file (optional)')
    parser.add_argument('-r', '--sample-rate', type=int, default=44100,
                        help='Sample rate for the output WAV file (default: 44100 Hz)')
    parser.add_argument('-c', '--channels', type=int, default=2,
                        help='Number of audio channels (default: 2)')
    parser.add_argument('-b', '--bits', type=int, default=16,
                        help='Bit depth (default: 16 bits)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Perform the conversion
        output_path = convert_mp3_to_wav(
            args.input_file,
            args.output,
            args.sample_rate,
            args.channels,
            args.bits
        )
        print(f"Conversion successful! WAV file saved at: {output_path}")
    except Exception as e:
        print(f"Error converting file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
