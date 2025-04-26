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

def process_directory(input_dir, output_dir=None, sample_rate=44100, channels=2, bits=16):
    """
    Process all MP3 files in a directory and convert them to WAV format
    
    Args:
        input_dir (str): Path to the input directory containing MP3 files
        output_dir (str, optional): Path to save the output WAV files. If not provided,
                                    will save in the same directory as the input files
        sample_rate (int, optional): Sample rate for the output WAV files. Defaults to 44100 Hz.
        channels (int, optional): Number of audio channels. Defaults to 2 (stereo).
        bits (int, optional): Bit depth. Defaults to 16 bits.
    
    Returns:
        list: Paths to the created WAV files
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    converted_files = []
    errors = []
    
    # Get all MP3 files in the directory
    mp3_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp3')]
    
    if not mp3_files:
        print(f"No MP3 files found in directory: {input_dir}")
        return []
    
    # Process each MP3 file
    for mp3_file in mp3_files:
        input_path = os.path.join(input_dir, mp3_file)
        
        if output_dir:
            output_filename = os.path.splitext(mp3_file)[0] + ".wav"
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_path = None  # Let convert_mp3_to_wav determine the output path
        
        try:
            wav_path = convert_mp3_to_wav(
                input_path,
                output_path,
                sample_rate,
                channels,
                bits
            )
            converted_files.append(wav_path)
            print(f"Converted: {mp3_file} -> {os.path.basename(wav_path)}")
        except Exception as e:
            errors.append((mp3_file, str(e)))
            print(f"Error converting {mp3_file}: {e}")
    
    # Report summary
    print(f"\nConversion Summary:")
    print(f"  Total MP3 files found: {len(mp3_files)}")
    print(f"  Successfully converted: {len(converted_files)}")
    print(f"  Failed conversions: {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for file, error in errors:
            print(f"  {file}: {error}")
    
    return converted_files

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Convert all MP3 files in a directory to WAV format')
    
    # Add arguments
    parser.add_argument('input_dir', help='Path to the directory containing MP3 files')
    parser.add_argument('-o', '--output-dir', help='Directory to save the output WAV files (optional)')
    parser.add_argument('-r', '--sample-rate', type=int, default=44100,
                        help='Sample rate for the output WAV files (default: 44100 Hz)')
    parser.add_argument('-c', '--channels', type=int, default=2,
                        help='Number of audio channels (default: 2)')
    parser.add_argument('-b', '--bits', type=int, default=16,
                        help='Bit depth (default: 16 bits)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Process the directory
        process_directory(
            args.input_dir,
            args.output_dir,
            args.sample_rate,
            args.channels,
            args.bits
        )
    except Exception as e:
        print(f"Error processing directory: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
