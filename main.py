from moviepy.editor import VideoFileClip

def run_evaluation(video_file, ground_truth_csv):
    ground_truth_data = read_ground_truth_from_csv(ground_truth_csv)
    video_clip = VideoFileClip(video_file)
    frame_list = [frame for frame in video_clip.iter_frames()]

    all_detected_vehicles = []
    all_ground_truth_vehicles = []

    for frame_number, frame in enumerate(frame_list):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        final_frame = pipeline(frame_rgb)

        ground_truth_vehicles = []
        for label in ground_truth_data:
            ground_truth_vehicles.extend(ground_truth_data[label])

        detected_vehicles = detect_vehicles(final_frame)
        all_detected_vehicles.extend(detected_vehicles)
        all_ground_truth_vehicles.extend(ground_truth_vehicles)

    tp, fp, fn = evaluate_detection(all_detected_vehicles, all_ground_truth_vehicles)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    output_file = 'output_video_with_detections.mp4'
    final_clip = VideoFileClip(video_file).fl_image(lambda img: pipeline(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
    final_clip.write_videofile(output_file, codec='libx264')

if __name__ == "__main__":
    input_video_path = "C:Users\tejas\downloads\vehicle_detection\input_video.mp4"
    fixed_video_path = "C:Users\tejas\downloads\vehicle_detection\fixed_video.mp4"
    ground_truth_csv_path = "C:Users\tejas\downloads\vehicle_detection\ground_truth_csv.csv"

    reencode_video(input_video_path, fixed_video_path)
    run_evaluation(fixed_video_path, ground_truth_csv_path)
