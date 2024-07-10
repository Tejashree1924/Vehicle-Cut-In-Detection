import torch
import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

model_name = "mono+stereo_1024x320"
download_model_if_doesnt_exist(model_name)
model_path = os.path.join("models", model_name)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

# Load pretrained model
encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.eval()

depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.eval()

def estimate_depth(frame):
    input_image = torch.from_numpy(frame).unsqueeze(0).float()
    features = encoder(input_image)
    outputs = depth_decoder(features)
    disp = outputs[("disp", 0)]
    scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
    return scaled_disp[0].cpu().numpy()

# Example usage
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    depth = estimate_depth(frame)
    cv2.imshow('Depth', depth)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

