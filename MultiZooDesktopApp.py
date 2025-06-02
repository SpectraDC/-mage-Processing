
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk # type: ignore
import torch # type: ignore
import torchvision.transforms as T # type: ignore
import timm # type: ignore

# Cihaz ve sınıf isimleri
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

# Modeli oluştur ve eğitilmiş ağırlıkları yükle
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load("C:/Users/Furkan/Desktop/YazLab2-3/best_swin_model.pth", map_location=device))
model.eval().to(device)

# Görsel işleme pipeline
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Tahmin yapma fonksiyonu
def predict_image(path):
    image = Image.open(path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()

# Görsel seçme ve tahmin etme
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        pred_class, confidence = predict_image(file_path)

        img = Image.open(file_path).resize((224, 224))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        result_var.set(f"Tahmin: {pred_class} ({confidence:.2%})")

# Tkinter arayüzü
root = tk.Tk()
root.title("MultiZoo Görüntü Sınıflandırma")
root.geometry("400x400")

btn = tk.Button(root, text="Görsel Seç", command=load_image)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
