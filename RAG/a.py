
import os
import argparse
from roop.core import batch_process_regular
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions
from roop.face_util import extract_face_images
from roop.capturer import get_video_frame, get_video_frame_total, get_image_frame
from roop.FaceSet import FaceSet
from roop.utilities import clean_dir, detect_fps, get_video_frame_total, get_video_frame, get_image_frame
from roop.globals import CFG

def main():
    parser = argparse.ArgumentParser(description="Face Swap Script")
    parser.add_argument("--input", required=True, help="Path to the input image or video")
    parser.add_argument("--target", required=True, help="Path to the target image or video")
    parser.add_argument("--output", required=True, help="Path to save the output")
    parser.add_argument("--swap_model", default="InSwapper 128", help="Face swap model to use")
    parser.add_argument("--enhancer", default="None", help="Post-processing enhancer")
    parser.add_argument("--face_detection", default="First found", help="Face detection mode")
    parser.add_argument("--face_distance", type=float, default=0.65, help="Max face similarity threshold")
    parser.add_argument("--blend_ratio", type=float, default=0.65, help="Blend ratio for post-processing")
    parser.add_argument("--no_face_action", default="Use untouched original frame", help="Action when no face is detected")
    parser.add_argument("--vr_mode", action="store_true", help="Enable VR mode")
    parser.add_argument("--autorotate", action="store_true", help="Auto rotate horizontal faces")
    parser.add_argument("--num_swap_steps", type=int, default=1, help="Number of swapping steps")
    parser.add_argument("--upsample", default="128px", help="Subsample upscale to")
    args = parser.parse_args()

    

    # Extract faces from input and target
    input_faces = extract_face_images(args.input, (False, 0))
    target_faces = extract_face_images(args.target, (False, 0))

    if not input_faces or not target_faces:
        print("No faces detected in input or target!")
        return

    # Create FaceSet for input faces
    input_face_set = FaceSet()
    for face in input_faces:
        input_face_set.faces.append(face[0])
        input_face_set.ref_images.append(get_image_frame(args.input))

    # Create FaceSet for target faces
    target_face_set = FaceSet()
    for face in target_faces:
        target_face_set.faces.append(face[0])
        target_face_set.ref_images.append(get_image_frame(args.target))

    # Prepare process entries
    process_entry = ProcessEntry(args.target, 0, get_video_frame_total(args.target) if util.is_video(args.target) else 1, detect_fps(args.target) if util.is_video(args.target) else 0)
    process_entries = [process_entry]

    # Prepare process options
    options = ProcessOptions(
        args.swap_model,
        None,  # No mask engine for simplicity
        args.face_distance,
        args.blend_ratio,
        translate_swap_mode(args.face_detection),
        0,  # Face index
        "",  # No clip text
        None,  # No mask image
        args.num_swap_steps,
        int(args.upsample[:3]),
        False,  # No show face area
        False  # No restore original mouth
    )

    # Start processing
    batch_process_regular(args.swap_model, "File", process_entries, None, "", False, None, False, args.num_swap_steps, None, 0)

    print(f"Processing complete! Output saved to {args.output}")

def translate_swap_mode(mode):
    if mode == "Selected face":
        return "selected"
    elif mode == "First found":
        return "first"
    elif mode == "All input faces":
        return "all_input"
    elif mode == "All input faces (random)":
        return "all_random"
    elif mode == "All female":
        return "all_female"
    elif mode == "All male":
        return "all_male"
    return "all"

if __name__ == "__main__":
    main()
'''
import os
import argparse
import cv2
from roop.core import batch_process_regular
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions
from roop.face_util import extract_face_images
from roop.FaceSet import FaceSet
from roop.utilities import clean_dir, detect_fps, get_video_frame, get_image_frame
from roop.globals import CFG

def get_video_frame_total(video_path):
    """
    Get the total number of frames in a video using OpenCV.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def main():
    parser = argparse.ArgumentParser(description="Face Swap Script")
    parser.add_argument("--input", required=True, help="Path to the input image or video")
    parser.add_argument("--target", required=True, help="Path to the target image or video")
    parser.add_argument("--output", required=True, help="Path to save the output")
    parser.add_argument("--swap_model", default="InSwapper 128", help="Face swap model to use")
    parser.add_argument("--enhancer", default="None", help="Post-processing enhancer")
    parser.add_argument("--face_detection", default="First found", help="Face detection mode")
    parser.add_argument("--face_distance", type=float, default=0.65, help="Max face similarity threshold")
    parser.add_argument("--blend_ratio", type=float, default=0.65, help="Blend ratio for post-processing")
    parser.add_argument("--no_face_action", default="Use untouched original frame", help="Action when no face is detected")
    parser.add_argument("--vr_mode", action="store_true", help="Enable VR mode")
    parser.add_argument("--autorotate", action="store_true", help="Auto rotate horizontal faces")
    parser.add_argument("--num_swap_steps", type=int, default=1, help="Number of swapping steps")
    parser.add_argument("--upsample", default="128px", help="Subsample upscale to")
    args = parser.parse_args()

    # Prepare environment
    if CFG.clear_output:
        clean_dir(args.output)

    # Extract faces from input and target
    input_faces = extract_face_images(args.input, (False, 0))
    target_faces = extract_face_images(args.target, (False, 0))

    if not input_faces or not target_faces:
        print("No faces detected in input or target!")
        return

    # Create FaceSet for input faces
    input_face_set = FaceSet()
    for face in input_faces:
        input_face_set.faces.append(face[0])
        input_face_set.ref_images.append(get_image_frame(args.input))

    # Create FaceSet for target faces
    target_face_set = FaceSet()
    for face in target_faces:
        target_face_set.faces.append(face[0])
        target_face_set.ref_images.append(get_image_frame(args.target))

    # Prepare process entries
    process_entry = ProcessEntry(args.target, 0, get_video_frame_total(args.target) if util.is_video(args.target) else 1, detect_fps(args.target) if util.is_video(args.target) else 0)
    process_entries = [process_entry]

    # Prepare process options
    options = ProcessOptions(
        args.swap_model,
        None,  # No mask engine for simplicity
        args.face_distance,
        args.blend_ratio,
        translate_swap_mode(args.face_detection),
        0,  # Face index
        "",  # No clip text
        None,  # No mask image
        args.num_swap_steps,
        int(args.upsample[:3]),
        False,  # No show face area
        False  # No restore original mouth
    )

    # Start processing
    batch_process_regular(args.swap_model, "File", process_entries, None, "", False, None, False, args.num_swap_steps, None, 0)

    print(f"Processing complete! Output saved to {args.output}")

def translate_swap_mode(mode):
    if mode == "Selected face":
        return "selected"
    elif mode == "First found":
        return "first"
    elif mode == "All input faces":
        return "all_input"
    elif mode == "All input faces (random)":
        return "all_random"
    elif mode == "All female":
        return "all_female"
    elif mode == "All male":
        return "all_male"
    return "all"

if __name__ == "__main__":
    main()
'''    