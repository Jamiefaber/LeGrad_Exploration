import torch
import open_clip

class ViTWrapper:
    def __init__(self, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use open_clip exactly like in LeGrad docs
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, 
            pretrained=pretrained, 
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name=model_name)
        self.model.eval()
        self.image_size = 224  # You can change this to 448 as in their docs
        
    def encode_image(self, image_tensor, normalize=True):
        return self.model.encode_image(image_tensor.to(self.device), normalize=normalize)

    def encode_text(self, text_tensor, normalize=True):
        return self.model.encode_text(text_tensor.to(self.device), normalize=normalize)

    def preprocess_image(self, img):
        return self.preprocess(img).unsqueeze(0)

    def tokenize(self, prompt):
        if isinstance(prompt, str):
            prompt = [prompt]
        return self.tokenizer(prompt).to(self.device)