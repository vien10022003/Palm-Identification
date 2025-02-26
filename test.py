import cv2
import mediapipe as mp
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
from PIL import Image
from scipy.spatial.distance import cosine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Định nghĩa các phép biến đổi
transform = transforms.Compose([
    transforms.Grayscale(),  # Chuyển ảnh về grayscale
    transforms.Resize(224),  # Resize về kích thước 224x224
    transforms.ToTensor(),   # Chuyển thành Tensor
    transforms.Lambda(lambda x: x.repeat(3,1,1)),  # Lặp lại kênh để có 3 kênh (RGB giả lập)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Chuẩn hóa
])

def process_and_show_palm_crop(palm_crop):
    """Xử lý và hiển thị ảnh palm_crop theo pipeline biến đổi."""
    
    # Chuyển ảnh OpenCV (BGR) thành Grayscale
    palm_crop = cv2.cvtColor(palm_crop, cv2.COLOR_BGR2GRAY)
    
    # Chuyển từ NumPy array sang PIL Image để dùng với torchvision.transforms
    palm_crop_pil = Image.fromarray(palm_crop)

    # Áp dụng các phép biến đổi
    transformed_img = transform(palm_crop_pil)

    # Hiển thị ảnh
    imshow(transformed_img)

def imshow(inp, title=None): 
    """Hiển thị ảnh từ Tensor sau khi đã xử lý."""
    inp = inp.numpy().transpose((1, 2, 0))  # Chuyển về định dạng HxWxC
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # Đảo ngược quá trình chuẩn hóa
    inp = np.clip(inp, 0, 1)  # Giới hạn giá trị pixel trong khoảng [0,1]
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()



# Hàm chuyển ảnh thành feature vector
def extract_features(image, model):
    
    img = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        features = model(img)

    return features.cpu().numpy().flatten()

def register_new_person(image):
    # Chuyển ảnh OpenCV (BGR) thành Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Chuyển từ NumPy array sang PIL Image để dùng với torchvision.transforms
    image = Image.fromarray(image)
    
    # Đăng ký một lòng bàn tay mới
    new_person = "Nguyen Van B"

    # Trích xuất feature vector từ mô hình
    new_feature_vector = extract_features(image, model_ft)

    # # Lưu vector vào database hoặc file
    # registered_palm_data = {}  # Database tạm
    # registered_palm_data[new_person] = new_feature_vector
    registered_palm.append((new_person, new_feature_vector))
    
    # np.save("palm_database.npy", registered_palm_data)
    print(f"bpalm_database.npy: {new_feature_vector}")
    
registered_palm = []

def load_model(model_path):
    model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Tạo mô hình ResNet-18 trống
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 600)  # Cập nhật số lớp
    model_ft.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Tải trọng số
    model_ft.eval()  # Chuyển sang chế độ đánh giá
    return model_ft

# Tải mô hình từ đường dẫn
model_path = "resnet18_tongji_unfreezed.pt"
model_ft = load_model(model_path)

# registered_palm_data = np.load("palm_database.npy", allow_pickle=True).item()

def recognize_palm(image, model):
    # registered_palm_data = np.load("palm_database.npy", allow_pickle=True).item()
    # Chuyển ảnh OpenCV (BGR) thành Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Chuyển từ NumPy array sang PIL Image để dùng với torchvision.transforms
    image = Image.fromarray(image)
    
    # Trích xuất feature vector từ ảnh đầu vào
    feature_vector = extract_features(image, model)

    best_match = None
    best_score = float("inf")  # Khoảng cách nhỏ nhất

    # So sánh với các lòng bàn tay đã đăng ký
    for person, saved_vector in registered_palm:
        score = cosine(feature_vector, saved_vector)
        if score < best_score:  # Chọn người có khoảng cách nhỏ nhất
            best_score = score
            best_match = person
            
    print(f"best_score: {100-best_score*100}")
    print(f"best_match: {person}")
    return best_match if best_score < 0.4 else "Unknown"


def crop_hand_square(frame, hand_landmarks, w, h):
    # Lấy điểm chân ngón trỏ, chân ngón út và cổ tay
    wrist = hand_landmarks.landmark[0]  # Cổ tay
    index_base = hand_landmarks.landmark[5]  # Chân ngón trỏ
    pinky_base = hand_landmarks.landmark[17]  # Chân ngón út

    # Chuyển đổi tọa độ từ normalized (0-1) sang pixel
    x1, y1 = int(index_base.x * w), int(index_base.y * h)
    x2, y2 = int(pinky_base.x * w), int(pinky_base.y * h)
    x_wrist, y_wrist = int(wrist.x * w), int(wrist.y * h)

    # Tính độ dài cạnh hình vuông
    side_length = int(np.linalg.norm([x2 - x1, y2 - y1]))  

    # Tính vector hướng từ ngón trỏ đến ngón út
    dx, dy = x2 - x1, y2 - y1

    # Tính vector vuông góc
    perp_dx, perp_dy = -dy, dx  

    # Kiểm tra hướng của cổ tay để xác định chiều vuông góc đúng
    wrist_side = (x_wrist - x1) * dy - (y_wrist - y1) * dx  # Tích chéo
    if wrist_side >= 0:
        perp_dx, perp_dy = -perp_dx, -perp_dy  # Đảo hướng nếu cần

    # Chuẩn hóa vector để có độ dài bằng side_length
    norm_factor = side_length / np.linalg.norm([perp_dx, perp_dy])
    perp_dx, perp_dy = int(perp_dx * norm_factor), int(perp_dy * norm_factor)

    # Tính 4 điểm của hình vuông
    pts_src = np.array([
        [x1, y1],  # Điểm 1: Chân ngón trỏ
        [x2, y2],  # Điểm 2: Chân ngón út
        [x2 + perp_dx, y2 + perp_dy],  # Điểm 3: Kéo vuông góc đúng hướng
        [x1 + perp_dx, y1 + perp_dy]   # Điểm 4: Kéo vuông góc đúng hướng
    ], dtype=np.float32)
    
    # Tạo điểm đích (hình vuông chuẩn hóa)
    pts_dst = np.array([
        [0, 0], [side_length, 0], [side_length, side_length], [0, side_length]
    ], dtype=np.float32)

    # Tính toán ma trận biến đổi phối cảnh
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Cắt ảnh theo hình vuông đã xoay
    palm_crop = cv2.warpPerspective(frame, matrix, (side_length, side_length))

    # Vẽ hình vuông lên ảnh gốc (debug)
    cv2.polylines(frame, [pts_src.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Palm Rotated", palm_crop)
    return palm_crop, frame

# Khởi tạo mô hình nhận diện bàn tay của MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Mở camera
cap = cv2.VideoCapture(0)
        
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Lật ảnh để không bị ngược
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Chuyển ảnh sang RGB vì MediaPipe yêu cầu
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Dự đoán bàn tay
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                palm_crop, frame_debug = crop_hand_square(frame, hand_landmarks, w, h)

                if cv2.waitKey(1) & 0xFF == ord('e'):
                    if palm_crop.shape[0] > 0 and palm_crop.shape[1] > 0:
                    # cv2.imshow("Palm Crop", image_np)
                        process_and_show_palm_crop(palm_crop)
                   
                if cv2.waitKey(1) & 0xFF == ord('r'): 
                    if palm_crop.shape[0] > 0 and palm_crop.shape[1] > 0:   
                        register_new_person(palm_crop)
                        print(f"đăng kí thành công")
                if cv2.waitKey(1) & 0xFF == ord('t'): 
                    if palm_crop.shape[0] > 0 and palm_crop.shape[1] > 0:   
                        predicted_person = recognize_palm(palm_crop, model_ft)
                        print(f"Ảnh thuộc về: {predicted_person}")
                    
                # Hiển thị vùng cắt nếu hợp lệ
                # if palm_crop.shape[0] > 0 and palm_crop.shape[1] > 0:
                #     # cv2.imshow("Palm Crop", image_np)
                #     process_and_show_palm_crop(palm_crop)

        # Hiển thị ảnh từ camera
        cv2.imshow("Camera", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()